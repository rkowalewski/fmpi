#ifndef FMPI_MPI_DISPATCHER_HPP
#define FMPI_MPI_DISPATCHER_HPP

#include <mpi.h>

#include <deque>
#include <mutex>
#include <numeric>
#include <thread>

#include <tlx/container/ring_buffer.hpp>
#include <tlx/container/simple_vector.hpp>

#include <fmpi/Constants.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/Utils.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
#include <fmpi/detail/Capture.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/Request.hpp>
#include <fmpi/mpi/TypeMapper.hpp>

#include <fmpi/container/BoundedBuffer.hpp>
#include <unordered_map>

namespace fmpi {

enum class status
{
  pending,
  running,
  resolved,
  rejected
};

struct Ticket {
  uint32_t id{};
};

template <typename R, typename... Args>
class MpiTask {
  /// We want to make sure that our function fits into a single cache line
  static constexpr std::size_t overhead = sizeof(FixedFunction<void(), 0>);

  static constexpr std::size_t size =
      std::min<std::size_t>(128, 64 - overhead);

  fmpi::FixedFunction<R(Args..., MPI_Request*), size> f_;

  std::tuple<Args...> args_;
  status              status_{status::pending};

 public:
  MpiTask() = default;

  template <typename U>
  MpiTask(U&& u, std::tuple<Args...>&& args)
    : f_(std::forward<U>(u))
    , args_(std::move(args)) {
    FMPI_DBG(sizeof(f_));
    FMPI_DBG(sizeof(args_));
    FMPI_DBG(sizeof(*this));

    // static_assert(
    //    sizeof(std::aligned_storage<6 * sizeof(void*)>::type) <=
    //        sizeof(decltype(*this)),
    //    "");
  }

  R operator()(MPI_Request* req) {
    FMPI_ASSERT(status_ == status::pending);
    auto params = std::tuple_cat(std::move(args_), std::make_tuple(req));
    auto ret    = std::apply(f_, std::move(params));

    status_ = status::running;

    return ret;
  };

  status status() const noexcept {
    return status_;
  }
};  // namespace fmpi

template <typename F, typename... Args>
auto captureMpiTask(F&& f, Args&&... args)
    -> MpiTask<int, std::decay_t<Args>...> {
  return {std::forward<F>(f), std::make_tuple(std::forward<Args>(args)...)};
}

template <class T>
class CommDispatcher {
  using result_t = int;
  using buffer_t = fmpi::Span<T>;
  using rank_t   = mpi::Rank;
  using tag_t    = int;

  /// Task Signature
  using task_t = MpiTask<result_t, buffer_t, rank_t, tag_t>;

  /// Callback Signature
  using callback_t = FixedFunction<void(Ticket, MPI_Status, buffer_t), 64>;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  /// request type
  enum class request_type : uint8_t
  {
    IRECV = 0,
    ISEND,

    SENTINEL
  };

  static constexpr auto n_types = to_underlying(request_type::SENTINEL);

  static_assert(n_types == 2, "only two request types supported for now");

  /// Workload Item keeping a task, done callback and a ticket
  struct alignas(CACHE_ALIGNMENT) workload_item {
    task_t     task{};
    callback_t callback{};
    buffer_t   data{};
    Ticket     ticket{};
  };

  // All communication requests need to be in a single MPI context.
  mpi::Context const* ctx_{};

  // Uniq Task ID
  uint32_t seqCounter_{};  // counter for uniq ticket ids

  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  // Task Lists: One for each type
  std::array<std::list<workload_item>, n_types> queues_;
  // total over all tasks
  std::size_t ntasks_{};
  std::size_t busy_{};

  // Mutex to protect work sharing variables
  std::mutex mutex_;
  // Condition to signal empty tasks
  std::condition_variable cv_tasks_;
  // Condition to signal a finished task
  std::condition_variable cv_finished_;

  //////////////////////////////////
  // Sliding Window Worker Thread //
  //////////////////////////////////

  // Window Size
  std::size_t const winsz_;
  // Requests in flight for each type, currently: winsz / n_types
  std::size_t const reqs_in_flight_;
  // Pending tasks: explicitly use a normal vector to default construct
  // elements: explicitly use a normal vector to default construct elements
  tlx::SimpleVector<workload_item> pending_;
  // MPI_Requests
  simple_vector<MPI_Request> mpi_reqs_;
  // MPI_Statuses - only for receive requests
  simple_vector<MPI_Status> mpi_statuses_;
  // indices of request slots in the sliding window
  simple_vector<int> indices_;
  // free slots for each request type
  std::array<tlx::RingBuffer<int>, n_types> req_slots_;

  std::thread      consumer_;
  std::atomic_bool running_;

 public:
  CommDispatcher(mpi::Context const& ctx, std::size_t winsz);

  ~CommDispatcher();

  template <class Callback>
  Ticket postAsyncSend(
      buffer_t buffer, rank_t dest, tag_t tag, Callback&& cb);

  template <class Callback>
  Ticket postAsyncRecv(
      std::size_t count, rank_t source, tag_t tag, Callback&& cb);

  template <class Callback>
  Ticket postAsyncRecv(
      buffer_t buffer, rank_t source, tag_t tag, Callback&& cb);

  void loop_until_done() {
    std::unique_lock<std::mutex> lk{mutex_};
    cv_finished_.wait(lk, [this]() { return ntasks_ == 0 && busy_ == 0; });
  }

  void start_worker();
  void stop_worker();

 private:
  Ticket enqueue_task(
      request_type type,
      task_t&&     task,
      callback_t&& callback,
      buffer_t     buffer);

  std::size_t process_requests();

  bool has_pending_requests() const noexcept {
    return std::any_of(
        std::begin(mpi_reqs_), std::end(mpi_reqs_), [](auto const& req) {
          return req != MPI_REQUEST_NULL;
        });
  }

  void worker();
};

template <class T>
CommDispatcher<T>::CommDispatcher(mpi::Context const& ctx, std::size_t winsz)
  : ctx_(&ctx)
  , winsz_(winsz)
  , reqs_in_flight_(winsz_ / n_types)
  , pending_(winsz_)
  , mpi_reqs_(winsz_)
  , mpi_statuses_(winsz_)
  , indices_(winsz_)
  , running_(true) {
  // initialize arrays to track relevant information...
  std::uninitialized_fill(
      std::begin(mpi_reqs_), std::end(mpi_reqs_), MPI_REQUEST_NULL);

  std::uninitialized_fill(
      std::begin(req_slots_),
      std::end(req_slots_),
      tlx::RingBuffer<int>{reqs_in_flight_});

  for (auto&& i : range(req_slots_.size())) {
    std::generate_n(
        std::front_inserter(req_slots_[i]),
        reqs_in_flight_,
        [n = i * reqs_in_flight_]() mutable { return n++; });
  }

  FMPI_DBG(sizeof(workload_item));

  FMPI_ASSERT((winsz % n_types) == 0);
}

template <class T>
CommDispatcher<T>::~CommDispatcher() {
  stop_worker();
  consumer_.join();
}

template <class T>
void CommDispatcher<T>::stop_worker() {
  running_.store(false, std::memory_order_relaxed);
}

template <class T>
void CommDispatcher<T>::start_worker() {
  consumer_ = std::thread(&CommDispatcher::worker, this);
}

template <class T>
template <class Callback>
Ticket CommDispatcher<T>::postAsyncRecv(
    buffer_t buffer, rank_t source, tag_t tag, Callback&& cb) {
  auto const* ctx = ctx_;

  auto task = fmpi::captureMpiTask(
      [ctx](auto buffer, auto peer, auto tag, MPI_Request* req) {
        return mpi::irecv(buffer.data(), buffer.size(), peer, tag, *ctx, req);
      },
      buffer,
      source,
      tag);

  return enqueue_task(
      request_type::IRECV,
      std::move(task),
      callback_t{std::forward<Callback>(cb)},
      buffer);
}

template <class T>
template <class Callback>
Ticket CommDispatcher<T>::postAsyncSend(
    buffer_t buffer, rank_t source, tag_t tag, Callback&& cb) {
  FMPI_ASSERT(!buffer.empty());

  auto const* ctx = ctx_;

  auto task = fmpi::captureMpiTask(
      [ctx](auto buffer, auto peer, auto tag, MPI_Request* req) {
        return mpi::isend(
            buffer.begin(), buffer.size(), peer, tag, *ctx, req);
      },
      buffer,
      source,
      tag);

  return enqueue_task(
      request_type::ISEND,
      std::move(task),
      callback_t{std::forward<Callback>(cb)},
      buffer);
}

template <class T>
Ticket CommDispatcher<T>::enqueue_task(
    request_type type, task_t&& task, callback_t&& callback, buffer_t data) {
  Ticket ticket;
  {
    // 1) Acquire lock
    std::lock_guard<std::mutex> lock(mutex_);

    // 2) Push task
    ticket = Ticket{seqCounter_++};

    ++ntasks_;

    queues_[to_underlying(type)].push_front(
        workload_item{std::move(task), std::move(callback), data, ticket});
  }  // 3) Release lock

  cv_tasks_.notify_one();

  return ticket;
}

template <class T>
void CommDispatcher<T>::worker() {
  std::vector<int> new_reqs;
  new_reqs.reserve(winsz_);

  // loop until termination
  while (running_.load(std::memory_order_relaxed)) {
    // 1) we first process pending requests...
    auto const nCompleted = has_pending_requests() ? process_requests() : 0;

    cv_finished_.notify_one();

    // 2) Schedule new requests or at least wait for new requests...
    {
      // acquire the lock
      std::unique_lock<std::mutex> lk(mutex_);

      busy_ -= nCompleted;

      if (!ntasks_) {
        // wait for new tasks for a maximum of 10ms
        // If a timeout occurs, unlock and
        auto const ten_ms = std::chrono::milliseconds(10);
        if (!cv_tasks_.wait_for(
                lk, ten_ms, [this]() { return ntasks_ > 0; })) {
          // lock is again released...
          continue;
        };
      }

      for (auto&& q : range(n_types)) {
        auto const n = std::min(req_slots_[q].size(), queues_[q].size());
        for (auto&& unused : range(n)) {
          std::ignore = unused;

          auto task = std::move(queues_[q].back());
          queues_[q].pop_back();

          // get a free slot
          auto const slot = req_slots_[q].back();
          req_slots_[q].pop_back();

          pending_[slot] = std::move(task);
          new_reqs.push_back(slot);
        }
      }
      ntasks_ -= new_reqs.size();
      busy_ += new_reqs.size();
    }  // release the lock

    // 3) execute new requests. Do this after releasing the lock.
    for (auto&& slot : new_reqs) {
      auto* mpi_req = &mpi_reqs_[slot];
      // initiate the task
      pending_[slot].task(mpi_req);
    }

    new_reqs.clear();
  }
}  // namespace fmpi

template <class T>
std::size_t CommDispatcher<T>::process_requests() {
  int* last;

  auto mpi_ret = mpi::testsome(
      &*std::begin(mpi_reqs_),
      &*std::end(mpi_reqs_),
      indices_.data(),
      mpi_statuses_.data(),
      last);

  FMPI_CHECK(mpi_ret == MPI_SUCCESS);

  auto const nCompleted = std::distance(&*std::begin(indices_), last);

  // left half of reqs array array are receives
  // right half sends
  auto* fstSent =
      std::partition(&*std::begin(indices_), last, [this](auto const& req) {
        return req < static_cast<int>(reqs_in_flight_);
      });

  std::copy(
      &*std::begin(indices_),
      fstSent,
      std::front_inserter(req_slots_[to_underlying(request_type::IRECV)]));

  std::copy(
      fstSent,
      last,
      std::front_inserter(req_slots_[to_underlying(request_type::ISEND)]));

  for (auto it = &*std::begin(indices_); it < fstSent; ++it) {
    auto& processed = pending_[*it];

    // we explcitly set the error field to propagate succeeded MPI calls.
    // MPI does not do that, unfortunately.
    mpi_statuses_[*it].MPI_ERROR =
        (mpi_ret == MPI_SUCCESS) ? MPI_SUCCESS : mpi_statuses_[*it].MPI_ERROR;

    processed.callback(processed.ticket, mpi_statuses_[*it], processed.data);
  }

  return nCompleted;
}

}  // namespace fmpi
#endif
