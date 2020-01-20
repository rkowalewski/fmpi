#ifndef FMPI_MPI_DISPATCHER_HPP
#define FMPI_MPI_DISPATCHER_HPP

#include <mpi.h>

#include <atomic>
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

template <class T>
class CommDispatcher {
  using result_t = int;
  using buffer_t = fmpi::Span<T>;
  using rank_t   = mpi::Rank;
  using tag_t    = int;

  /// Task Signature
  using task_t = Function<int(MPI_Request*)>;

  /// Callback Signature
  using callback_t = Function<void(Ticket, MPI_Status, buffer_t)>;

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
  struct workload_item {
    task_t     task{};
    callback_t callback{};
    buffer_t   data{};
    Ticket     ticket{};
    status     state{status::pending};

    workload_item() = default;

    workload_item(task_t&& f, callback_t&& c, buffer_t d, Ticket t)
      : task(std::move(f))
      , callback(std::move(c))
      , data(d)
      , ticket(t) {
    }
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

  std::thread      thread_;
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

  void pinToCore(int coreId);

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
inline CommDispatcher<T>::CommDispatcher(
    mpi::Context const& ctx, std::size_t winsz)
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

  FMPI_ASSERT((winsz % n_types) == 0);
}

template <class T>
inline CommDispatcher<T>::~CommDispatcher() {
  stop_worker();
  thread_.join();
}

template <class T>
inline void CommDispatcher<T>::stop_worker() {
  running_.store(false, std::memory_order_relaxed);
}

template <class T>
void CommDispatcher<T>::start_worker() {
  thread_ = std::thread(&CommDispatcher::worker, this);
}

template <class T>
template <class Callback>
inline Ticket CommDispatcher<T>::postAsyncRecv(
    buffer_t buffer, rank_t source, tag_t tag, Callback&& cb) {
  auto const* ctx = ctx_;

  auto task = fmpi::makeCapture<int>(
      [ctx, buffer, peer = source, tag](MPI_Request* req) {
        return mpi::irecv(buffer.data(), buffer.size(), peer, tag, *ctx, req);
      });

  auto callback = fmpi::makeCapture<void>(std::forward<Callback>(cb));

  return enqueue_task(
      request_type::IRECV, std::move(task), std::move(callback), buffer);
}

template <class T>
template <class Callback>
inline Ticket CommDispatcher<T>::postAsyncSend(
    buffer_t buffer, rank_t dest, tag_t tag, Callback&& cb) {
  FMPI_ASSERT(!buffer.empty());

  std::ignore = cb;

  auto const* ctx = ctx_;

  auto task = fmpi::makeCapture<int>([ctx, buffer, peer = dest, tag](
                                         MPI_Request* req) {
    return mpi::isend(buffer.begin(), buffer.size(), peer, tag, *ctx, req);
  });

  auto callback = fmpi::makeCapture<void>(std::forward<Callback>(cb));

  return enqueue_task(
      request_type::ISEND, std::move(task), std::move(callback), buffer);
}

template <class T>
inline Ticket CommDispatcher<T>::enqueue_task(
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
inline void CommDispatcher<T>::worker() {
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
        constexpr auto ten_ms = std::chrono::milliseconds(10);
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
      auto prev = std::exchange(pending_[slot].state, status::running);
      FMPI_ASSERT(prev == status::pending);
    }

    new_reqs.clear();
  }
}

template <class T>
inline std::size_t CommDispatcher<T>::process_requests() {
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

    auto const prev = std::exchange(
        processed.state,
        (mpi_ret == MPI_SUCCESS) ? status::resolved : status::rejected);

    FMPI_ASSERT(prev == status::running);

    // TODO error handling if some requests fail?

    processed.callback(processed.ticket, mpi_statuses_[*it], processed.data);
  }

  return nCompleted;
}

template <class T>
inline void CommDispatcher<T>::pinToCore(int coreId) {
  int cpuSetSize = sizeof(cpu_set_t);
  if (coreId >= 0 && (coreId <= cpuSetSize * 8)) {
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(coreId, &cpuSet);
    pthread_setaffinity_np(thread_.native_handle(), cpuSetSize, &cpuSet);
  }
}

}  // namespace fmpi
#endif
