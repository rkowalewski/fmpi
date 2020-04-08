#ifndef FMPI_DISPATCHER_HPP
#define FMPI_DISPATCHER_HPP

#include <mpi.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <new>
#include <numeric>
#include <queue>
#include <thread>

#include <rtlx/Enum.hpp>
#include <rtlx/Timer.hpp>

#include <tlx/container/ring_buffer.hpp>
#include <tlx/container/simple_vector.hpp>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/concurrency/UnlockGuard.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
#include <fmpi/container/InstrusiveList.hpp>
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

std::ostream& operator<<(std::ostream& os, fmpi::Ticket const& ticket);

constexpr bool operator==(Ticket const& l, Ticket const& r) noexcept {
  return l.id == r.id;
}

class Message {
  struct Envelope {
    mpi::Comm comm{MPI_COMM_NULL};
    mpi::Rank peer{};
    mpi::Tag  tag{};

    Envelope() = default;

    Envelope(mpi::Rank peer_, mpi::Tag tag_, mpi::Comm comm_)
      : comm(comm_)
      , peer(peer_)
      , tag(tag_) {
    }
  };

 public:
  Message() = default;

  Message(mpi::Rank peer, mpi::Tag tag, mpi::Context const& comm) noexcept
    : envelope_(peer, tag, comm.mpiComm()) {
  }

  template <class T>
  Message(
      Span<T>             span,
      mpi::Rank           peer,
      mpi::Tag            tag,
      mpi::Context const& comm) noexcept
    : envelope_(peer, tag, comm.mpiComm()) {
    set_buffer(span.data(), span.size());
  }

  Message(const Message&) = delete;
  Message& operator=(Message const&) = delete;

  Message(Message&&) = default;
  Message& operator=(Message&&) = default;

  void set_buffer(void* buf, std::size_t count, MPI_Datatype type) {
    buf_   = buf;
    count_ = count;
    type_  = type;
  }

  template <class T>
  void set_buffer(T* buf, std::size_t count) {
    set_buffer(buf, count, mpi::type_mapper<T>::type());
  }

  template <class T>
  void set_buffer(Span<T> buf) {
    set_buffer(buf.data(), buf.size(), mpi::type_mapper<T>::type());
  }

  void* writable_buffer() noexcept {
    return buf_;
  }

  const void* readable_buffer() const noexcept {
    return buf_;
  }

  MPI_Datatype type() const noexcept {
    return type_;
  }

  std::size_t count() const noexcept {
    return count_;
  }

  mpi::Rank peer() const noexcept {
    return envelope_.peer;
  }

  mpi::Comm comm() const noexcept {
    return envelope_.comm;
  }

  mpi::Tag tag() const noexcept {
    return envelope_.tag;
  }

 private:
  void*        buf_{};
  std::size_t  count_{};
  MPI_Datatype type_{};
  Envelope     envelope_{};
};

template <mpi::reqsome_op testReqs = mpi::testsome>
class CommDispatcher {
  using Timer    = rtlx::Timer<>;
  using duration = typename Timer::duration;

  struct Task {
    Task() = default;

    Task(Message&& m, request_type t)
      : message(std::move(m))
      , type(t) {
    }

    Message      message{};
    request_type type{};
  };

 public:
  struct Statistics {
    // modified internally, read externally
    std::size_t completed{};
    std::size_t iterations{};
    duration    dispatch_time{};
    duration    queue_time{};
    duration    completion_time{};
  };

 private:
  static constexpr uint16_t default_task_capacity = 1000;
  /// Task Signature
  using task_func = Function<int(MPI_Request*, Ticket)>;
  /// Callback Signature
  using callback_func = Function<void(MPI_Status, Ticket)>;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  static constexpr auto n_types = rtlx::to_underlying(request_type::SENTINEL);

  static_assert(n_types == 2, "only two request types supported for now");

  using task_allocator = HeapAllocator<Task, false>;
  using task_list  = std::list<Task, ContiguousPoolAllocator<Task, false>>;
  using task_queue = std::queue<Task, task_list>;

  using signal   = Function<int(Message&, MPI_Request&)>;
  using callback = Function<void(Message&, MPI_Status const&)>;

  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  // Task Lists: One for each type
  alignas(kCacheLineAlignment) std::array<task_queue, n_types> queues_;

  task_allocator alloc_;

  Statistics stats_{};

  // Mutex to protect work sharing variables
  mutable std::mutex mutex_;
  // Condition to signal empty tasks
  std::condition_variable cv_tasks_;
  // Condition to signal a finished task
  std::condition_variable cv_finished_;

  // Termination Flag
  bool terminate_{false};

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
  std::array<std::list<signal>, n_types>    signals_;
  std::array<std::list<callback>, n_types>  callbacks_;

  std::thread thread_;

  // Uniq Task ID
  uint32_t seqCounter_{};  // counter for uniq ticket ids
 public:
  explicit CommDispatcher(std::size_t winsz);

  ~CommDispatcher();

  void postAsync(request_type qid, Message&& message);

  template <class F, class... Args>
  void register_signal(request_type type, F&& sig, Args&&... args);

  template <class F, class... Args>
  void register_callback(request_type type, F&& cb, Args&&... args);

  void loop_until_done();

  void start_worker();
  void stop_worker();

  void pinToCore(int coreId);

  Statistics stats() const;

 private:
  std::size_t queue_count() const noexcept;

  std::size_t req_count() const noexcept;

  std::size_t req_capacity() const noexcept;

  [[nodiscard]] bool has_active_requests() const noexcept;

  std::size_t progress_requests();

  // !!! CAUTION: Never Call this without holding the mutex !!!
  bool is_all_done() const noexcept;

  void worker();
};  // namespace fmpi

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(std::size_t winsz)
  : alloc_(task_allocator{default_task_capacity})
  , winsz_(winsz)
  , reqs_in_flight_(winsz_ / n_types)
  , pending_(winsz_)
  , mpi_reqs_(winsz_)
  , mpi_statuses_(winsz_)
  , indices_(winsz_) {
  for (auto&& i : range(queues_.size())) {
    queues_[i] = std::move(task_queue{alloc_});
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
  stop_worker();

  thread_.join();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::stop_worker() {
  std::unique_lock<std::mutex> lk{mutex_};
  terminate_ = true;
  cv_tasks_.notify_all();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::start_worker() {
  thread_ = std::thread(&CommDispatcher::worker, this);
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::worker() {
  std::vector<int> new_reqs;
  new_reqs.reserve(winsz_);

  using lock     = std::unique_lock<std::mutex>;
  auto condition = [this]() { return queue_count() || terminate_; };

  // TODO: what is the best sleep_interval?
  using namespace std::chrono_literals;
  constexpr auto sleep_interval = 1ms;

  lock lk{mutex_};

  // loop until termination
  do {
    duration queue_time{};
    duration dispatch_time{};
    duration completion_time{};

    {
      Timer tq{queue_time};

      if (req_count() > 0) {
        // If we have pending requests we wait only a small time interval to
        // check if there are new messages. Regardless whether there are new
        // messages, this interval is short enough to progress pending
        // requests
        cv_tasks_.wait_for(lk, sleep_interval, condition);
      } else {
        // If there are no pending requests we wait until there are some
        // available
        cv_tasks_.wait(lk, condition);
      }

      if (queue_count() && req_capacity()) {
        for (auto&& q : range(n_types)) {
          auto const n = std::min(req_slots_[q].size(), queues_[q].size());
          for (auto&& unused : range(n)) {
            std::ignore = unused;

            auto task = std::move(queues_[q].front());
            queues_[q].pop();

            // get a free slot
            auto const slot = req_slots_[q].back();
            req_slots_[q].pop_back();

            pending_[slot] = std::move(task);

            new_reqs.emplace_back(slot);
          }
        }
      }
    }

    bool progress = false;
    {
      UnlockGuard ulg{lk};

      {
        Timer{dispatch_time};
        // 3) execute new requests. Do this after releasing the lock.
        for (auto&& slot : new_reqs) {
          // initiate the task
          auto& task    = pending_[slot];
          auto& signals = signals_[rtlx::to_underlying(task.type)];
          for (auto&& sig : signals) {
            sig(task.message, mpi_reqs_[slot]);
          }
        }

        progress = !new_reqs.empty();
        new_reqs.clear();
      }

      {
        Timer{completion_time};
        // 1) we first process pending requests...
        progress = progress || (progress_requests() > 0);
      }
    }

    stats_.iterations++;
    stats_.completion_time += completion_time;
    stats_.dispatch_time += dispatch_time;
    stats_.queue_time += queue_time;

    if (is_all_done() || progress) {
      cv_finished_.notify_all();
    }
  } while (!terminate_);
}

template <mpi::reqsome_op testReqs>
void CommDispatcher<testReqs>::postAsync(
    request_type qid, Message&& message) {
  {
    // 1) Acquire lock
    std::lock_guard<std::mutex> lock(mutex_);

    queues_[rtlx::to_underlying(qid)].push(Task{std::move(message), qid});
  }

  // 3) signal
  cv_tasks_.notify_one();
}

template <typename mpi::reqsome_op testReqs>
inline std::size_t CommDispatcher<testReqs>::progress_requests() {
  int* last;

  auto const nreqs = req_count();
  FMPI_DBG(nreqs);

  if (!nreqs) {
    return 0U;
  }

  auto mpi_ret = testReqs(
      &*std::begin(mpi_reqs_),
      &*std::end(mpi_reqs_),
      indices_.data(),
      mpi_statuses_.data(),
      last);

  FMPI_CHECK_MPI(mpi_ret);

  auto const nCompleted = std::distance(&*std::begin(indices_), last);

  FMPI_DBG(nCompleted);

  // left half of reqs array array are receives
  // right half sends
  auto* fstSent =
      std::partition(&*std::begin(indices_), last, [this](auto const& req) {
        return req < static_cast<int>(reqs_in_flight_);
      });

  std::copy(
      &*std::begin(indices_),
      fstSent,
      std::front_inserter(
          req_slots_[rtlx::to_underlying(request_type::IRECV)]));

  std::copy(
      fstSent,
      last,
      std::front_inserter(
          req_slots_[rtlx::to_underlying(request_type::ISEND)]));

  for (auto it = &*std::begin(indices_); it < fstSent; ++it) {
    auto& processed = pending_[*it];

    // we explcitly set the error field to propagate succeeded MPI calls.
    // MPI does not do that, unfortunately.
    mpi_statuses_[*it].MPI_ERROR =
        (mpi_ret == MPI_SUCCESS) ? MPI_SUCCESS : mpi_statuses_[*it].MPI_ERROR;

    auto& callbacks = callbacks_[rtlx::to_underlying(processed.type)];
    for (auto&& cb : callbacks) {
      cb(processed.message, mpi_statuses_[*it]);
    }
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
  cv_finished_.wait(lk, [this]() { return is_all_done(); });
}

template <mpi::reqsome_op testReqs>
template <class F, class... Args>
inline void CommDispatcher<testReqs>::register_signal(
    request_type type, F&& sig, Args&&... args) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  signals_[slot].push_back(
      signal::make(std::forward<F>(sig), std::forward<Args...>(args)...));
}

template <mpi::reqsome_op testReqs>
template <class F, class... Args>
inline void CommDispatcher<testReqs>::register_callback(
    request_type type, F&& sig, Args&&... args) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  callbacks_[slot].push_back(
      callback::make(std::forward<F>(sig), std::forward<Args...>(args)...));
}

template <mpi::reqsome_op testReqs>
inline std::size_t CommDispatcher<testReqs>::queue_count() const noexcept {
  return std::accumulate(
      std::begin(queues_),
      std::end(queues_),
      0U,
      [](auto acc, auto const& queue) { return acc + queue.size(); });
}

template <mpi::reqsome_op testReqs>
inline std::size_t CommDispatcher<testReqs>::req_count() const noexcept {
  return std::count_if(
      std::begin(mpi_reqs_), std::end(mpi_reqs_), [](auto const& req) {
        return req != MPI_REQUEST_NULL;
      });
}

template <mpi::reqsome_op testReqs>
inline std::size_t CommDispatcher<testReqs>::req_capacity() const noexcept {
  return std::accumulate(
      std::begin(req_slots_),
      std::end(req_slots_),
      0U,
      [](auto acc, auto const& queue) { return acc + queue.size(); });
}

template <mpi::reqsome_op testReqs>
inline bool CommDispatcher<testReqs>::has_active_requests() const noexcept {
  return req_count() > 0;
}

template <mpi::reqsome_op testReqs>
inline bool CommDispatcher<testReqs>::is_all_done() const noexcept {
  return !queue_count() && !req_count();
}

}  // namespace fmpi
#endif
