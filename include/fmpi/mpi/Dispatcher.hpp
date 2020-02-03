#ifndef FMPI_MPI_DISPATCHER_HPP
#define FMPI_MPI_DISPATCHER_HPP

#include <mpi.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <numeric>
#include <thread>

#include <rtlx/Timer.hpp>

#include <tlx/container/ring_buffer.hpp>
#include <tlx/container/simple_vector.hpp>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/Utils.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
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

/// request type
enum class request_type : uint8_t
{
  IRECV = 0,
  ISEND,

  SENTINEL
};

struct Ticket {
  uint32_t id{};
};

std::ostream& operator<<(std::ostream& os, Ticket const& ticket) {
  os << "{ id : " << ticket.id << " }";
  return os;
}
constexpr bool operator==(Ticket const& l, Ticket const& r) noexcept {
  return l.id == r.id;
}

template <mpi::reqsome_op testReqs = mpi::testsome>
class CommDispatcher {
  using Timer    = rtlx::Timer<>;
  using duration = typename Timer::duration;

 public:
  struct Statistics {
    std::size_t ntasks{};
    std::size_t busy{};
    std::size_t completed{};
    std::size_t iterations{};
    duration    dispatch_time{};
    duration    queue_time{};
    duration    completion_time{};
  };

 private:
  static constexpr uint16_t default_task_capacity = 1000;
  /// Task Signature
  using task_t = Function<int(MPI_Request*, Ticket)>;

  /// Callback Signature
  using callback_t = Function<void(MPI_Status, Ticket)>;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  static constexpr auto n_types = to_underlying(request_type::SENTINEL);

  static_assert(n_types == 2, "only two request types supported for now");

  /// Workload Item keeping a task, done callback and a ticket
  struct alignas(kCacheLineAlignment) Task {
    task_t     task{};
    callback_t callback{};
    Ticket     ticket{};
    status     state{status::pending};

    Task() = default;

    Task(task_t&& f, callback_t&& c, Ticket t)
      : task(std::move(f))
      , callback(std::move(c))
      , ticket(t) {
    }
  };

  using queue_list_allocator = HeapAllocator<Task, false>;
  using queue_list = std::list<Task, ContiguousPoolAllocator<Task, false>>;
  // using queue_list = std::list<Task>;

  // Uniq Task ID
  uint32_t seqCounter_{};  // counter for uniq ticket ids

  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  queue_list_allocator alloc_;

  // Task Lists: One for each type
  std::array<queue_list, n_types> queues_;
  // total over all tasks
  // std::size_t ntasks_{};
  // std::size_t busy_{};

  // Mutex to protect work sharing variables
  mutable std::mutex mutex_;
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
  tlx::SimpleVector<Task> pending_;
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

  Statistics stats_{};

 public:
  explicit CommDispatcher(std::size_t winsz);

  ~CommDispatcher();

  Ticket postAsync(
      request_type                          qid,
      Function<int(MPI_Request*, Ticket)>&& task,
      Function<void(MPI_Status, Ticket)>&&  callback);

  void loop_until_done();

  void start_worker();
  void stop_worker();

  void pinToCore(int coreId);

  Statistics stats() const;

 private:
  Ticket do_enqueue(request_type type, task_t&& task, callback_t&& callback);

  std::size_t process_requests();

  [[nodiscard]] bool has_pending_requests() const noexcept {
    return std::any_of(
        std::begin(mpi_reqs_), std::end(mpi_reqs_), [](auto const& req) {
          return req != MPI_REQUEST_NULL;
        });
  }

  void worker();
};

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(std::size_t winsz)
  : alloc_(queue_list_allocator{default_task_capacity})
  , winsz_(winsz)
  , reqs_in_flight_(winsz_ / n_types)
  , pending_(winsz_)
  , mpi_reqs_(winsz_)
  , mpi_statuses_(winsz_)
  , indices_(winsz_)
  , running_(true) {
  for (auto&& i : range(queues_.size())) {
    queues_[i] = std::move(queue_list{alloc_});
  }

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

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::~CommDispatcher() {
  //loop_until_done();
  stop_worker();
  thread_.join();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::stop_worker() {
  running_.store(false, std::memory_order_relaxed);
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::start_worker() {
  thread_ = std::thread(&CommDispatcher::worker, this);
}

template <typename mpi::reqsome_op testReqs>
inline Ticket CommDispatcher<testReqs>::postAsync(
    request_type                          qid,
    Function<int(MPI_Request*, Ticket)>&& task,
    Function<void(MPI_Status, Ticket)>&&  callback) {
  return do_enqueue(qid, std::move(task), std::move(callback));
}

template <typename mpi::reqsome_op testReqs>
inline Ticket CommDispatcher<testReqs>::do_enqueue(
    request_type type, task_t&& task, callback_t&& callback) {
  Ticket ticket;
  {
    // 1) Acquire lock
    std::lock_guard<std::mutex> lock(mutex_);

    // 2) Push task
    ticket = Ticket{seqCounter_++};

    ++stats_.ntasks;

    queues_[to_underlying(type)].push_front(
        Task{std::move(task), std::move(callback), ticket});
  }  // 3) Release lock

  cv_tasks_.notify_one();

  return ticket;
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::worker() {
  std::vector<int> new_reqs;
  new_reqs.reserve(winsz_);

  duration dispatch_time{};
  duration completion_time{};
  duration queue_time{};

  auto inc_time = [](auto const&monotonic_time, auto& target) {
    FMPI_ASSERT(!(target > monotonic_time));
    auto const diff = monotonic_time - target;
    target += diff;
  };

  // loop until termination
  while (running_.load(std::memory_order_relaxed)) {
    std::size_t nCompleted;

    {
      Timer{completion_time};
      // 1) we first process pending requests...
      nCompleted = has_pending_requests() ? process_requests() : 0;

      cv_finished_.notify_one();
    }

    // 2) Schedule new requests or at least wait for new requests...
    {
      Timer{queue_time};

      // acquire the lock
      std::unique_lock<std::mutex> lk(mutex_);

      inc_time(completion_time, stats_.completion_time);
      inc_time(queue_time, stats_.queue_time);
      inc_time(dispatch_time, stats_.dispatch_time);

      stats_.iterations++;
      stats_.busy -= nCompleted;

      if (stats_.ntasks == 0u) {
        // wait for new tasks for a maximum of 10ms
        // If a timeout occurs, unlock and
        constexpr auto interval = std::chrono::microseconds(1);
        if (!cv_tasks_.wait_for(
                lk, interval, [this]() {return stats_.ntasks > 0; })) {
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
      stats_.ntasks -= new_reqs.size();
      stats_.busy += new_reqs.size();
    }  // release the lock

    {
      Timer{dispatch_time};
      // 3) execute new requests. Do this after releasing the lock.
      for (auto&& slot : new_reqs) {
        auto* mpi_req = &mpi_reqs_[slot];
        // initiate the task
        pending_[slot].task(mpi_req, pending_[slot].ticket);
        auto prev = std::exchange(pending_[slot].state, status::running);
        FMPI_ASSERT(prev == status::pending);
      }

      new_reqs.clear();
    }
  }
}

template <typename mpi::reqsome_op testReqs>
inline std::size_t CommDispatcher<testReqs>::process_requests() {
  int* last;

  auto mpi_ret = testReqs(
      &*std::begin(mpi_reqs_),
      &*std::end(mpi_reqs_),
      indices_.data(),
      mpi_statuses_.data(),
      last);

  FMPI_CHECK_MPI(mpi_ret);

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

    if (!processed.callback) continue;

    // we explcitly set the error field to propagate succeeded MPI calls.
    // MPI does not do that, unfortunately.
    mpi_statuses_[*it].MPI_ERROR =
        (mpi_ret == MPI_SUCCESS) ? MPI_SUCCESS : mpi_statuses_[*it].MPI_ERROR;

    auto const prev = std::exchange(
        processed.state,
        (mpi_ret == MPI_SUCCESS) ? status::resolved : status::rejected);

    FMPI_ASSERT(prev == status::running);

    // TODO(rkowalewski): error handling if some requests fail?

    processed.callback(mpi_statuses_[*it], processed.ticket);
  }

  return nCompleted;
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::pinToCore(int coreId) {
  auto const pin_success = pinThreadToCore(thread_, coreId);
  RTLX_ASSERT(pin_success);
}

template <mpi::reqsome_op testReqs>
typename CommDispatcher<testReqs>::Statistics
CommDispatcher<testReqs>::stats() const {
  std::lock_guard<std::mutex>(this->mutex_);
  return stats_;
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::loop_until_done() {
  std::unique_lock<std::mutex> lk{mutex_};
  if (!(stats_.ntasks == 0 && stats_.busy == 0)) {
    cv_finished_.wait(
        lk, [this]() { return stats_.ntasks == 0 && stats_.busy == 0; });
  }
}

}  // namespace fmpi
#endif
