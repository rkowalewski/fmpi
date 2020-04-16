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

#include <gsl/span>

#include <rtlx/Enum.hpp>
#include <rtlx/Timer.hpp>

#include <tlx/container/ring_buffer.hpp>
#include <tlx/container/simple_vector.hpp>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/concurrency/UnlockGuard.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
#include <fmpi/container/IntrusiveList.hpp>
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
enum class message_type : uint8_t
{
  IRECV = 0,
  ISEND,

  SENTINEL
};

class Message {
  struct Envelope {
    mpi::Comm comm{MPI_COMM_NULL};
    mpi::Rank peer{};
    mpi::Tag  tag{};

    Envelope() = default;

    Envelope(mpi::Rank peer_, mpi::Tag tag_, mpi::Comm comm_) noexcept
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
      gsl::span<T>        span,
      mpi::Rank           peer,
      mpi::Tag            tag,
      mpi::Context const& comm) noexcept
    : envelope_(peer, tag, comm.mpiComm()) {
    set_buffer(span.data(), span.size());
  }

  Message(const Message&) = default;
  Message& operator=(Message const&) = default;

  Message(Message&&) noexcept = default;
  Message& operator=(Message&&) noexcept = default;

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
  void set_buffer(gsl::span<T> buf) {
    set_buffer(buf.data(), buf.size(), mpi::type_mapper<T>::type());
  }

  void* writable_buffer() noexcept {
    return buf_;
  }

  [[nodiscard]] const void* readable_buffer() const noexcept {
    return buf_;
  }

  [[nodiscard]] MPI_Datatype type() const noexcept {
    return type_;
  }

  [[nodiscard]] std::size_t count() const noexcept {
    return count_;
  }

  [[nodiscard]] mpi::Rank peer() const noexcept {
    return envelope_.peer;
  }

  [[nodiscard]] mpi::Comm comm() const noexcept {
    return envelope_.comm;
  }

  [[nodiscard]] mpi::Tag tag() const noexcept {
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

    Task(Message m, message_type t)
      : message(m)
      , type(t) {
    }

    Message      message{};
    message_type type{};
  };

  class RequestWindow {};

 public:
  struct Statistics {
    // modified internally, read externally
    std::size_t iterations{};

    duration queue_time{};
    duration dispatch_time{};
    duration completion_time{};
    duration callback_time{};
  };

 private:
  static constexpr uint16_t default_task_capacity = 1000;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  static constexpr auto n_types = rtlx::to_underlying(message_type::SENTINEL);

  static_assert(n_types == 2, "only two request types supported for now");

  using contiguous_heap_allocator = HeapAllocator<Task, false>;
  using task_list  = std::list<Task, ContiguousPoolAllocator<Task, false>>;
  using task_queue = std::queue<Task, task_list>;

  using signal        = Function<int(Message&, MPI_Request&)>;
  using callback      = Function<void(Message&)>;
  using signal_list   = std::list<signal>;
  using callback_list = std::list<callback>;

  using signal_token   = typename signal_list::const_iterator;
  using callback_token = typename callback_list::const_iterator;

  using req_idx_t    = simple_vector<int>;
  using idx_ranges_t = std::array<int*, n_types>;

  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  // Task Lists: One for each type
  std::array<task_queue, n_types> queues_{};

  contiguous_heap_allocator alloc_;

  std::array<signal_list, n_types>   signals_{};
  std::array<callback_list, n_types> callbacks_{};

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
  std::size_t winsz_{};
  // Requests in flight for each type, currently: winsz / n_types
  std::size_t reqs_in_flight_{};
  // Pending tasks: explicitly use a normal vector to default construct
  tlx::SimpleVector<Task> pending_{};
  // MPI_Requests
  simple_vector<MPI_Request> mpi_reqs_{};
  // MPI_Statuses - only for receive requests
  simple_vector<MPI_Status> mpi_statuses_{};
  // indices of request slots in the sliding window
  req_idx_t indices_{};
  // free slots for each request type
  simple_vector<tlx::RingBuffer<int>> req_slots_{n_types};

  std::thread thread_;

 public:
  explicit CommDispatcher(std::size_t winsz);

  ~CommDispatcher();

  void dispatch(message_type qid, Message message);

  template <class F, class... Args>
  signal_token register_signal(
      message_type type, F&& callable, Args&&... args);

  template <class F, class... Args>
  callback_token register_callback(
      message_type type, F&& callable, Args&&... args);

  void loop_until_done();

  void start_worker();
  void stop_worker();

  void reset(std::size_t winsz);

  void pinToCore(int coreId);

  Statistics stats() const;

 private:
  std::size_t queue_count() const noexcept;

  std::size_t req_count() const noexcept;

  std::size_t req_capacity() const noexcept;

  [[nodiscard]] bool has_active_requests() const noexcept;

  idx_ranges_t progress_requests();

  // !!! CAUTION: Never Call this without holding the mutex !!!
  bool is_all_done() const noexcept;

  void worker();

  void discard_signals();

  void do_reset(std::size_t winsz);

  void trigger_callbacks(idx_ranges_t ranges);
};  // namespace fmpi

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(std::size_t winsz)
  : alloc_(contiguous_heap_allocator{default_task_capacity}) {
  do_reset(winsz);
}

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::~CommDispatcher() {
  stop_worker();
  thread_.join();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::stop_worker() {
  {
    std::lock_guard<std::mutex> lk{mutex_};
    terminate_ = true;
  }

  cv_tasks_.notify_all();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::start_worker() {
  thread_ = std::thread([this]() { worker(); });
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::worker() {
  std::vector<int> new_reqs;
  new_reqs.reserve(winsz_);

  using lock     = std::unique_lock<std::mutex>;
  auto condition = [this]() { return queue_count() || terminate_ || req_count(); };

  // TODO(rkowalewski): what is the best sleep_interval?

  using namespace std::chrono_literals;
  constexpr auto sleep_interval = 1us;

  lock lk{mutex_};

  // loop until termination
  do {
    duration queue_time{};
    duration dispatch_time{};
    duration completion_time{};
    duration callback_time{};

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
          auto const slots = req_slots_[q].size();
          auto const tasks = queues_[q].size();
          for (auto&& unused : range(std::min(slots, tasks))) {
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

    bool progress = !new_reqs.empty();
    {
      UnlockGuard ulg{lk};

      if (!new_reqs.empty()) {
        Timer{dispatch_time};
        // 3) execute new requests. Do this after releasing the lock.
        for (auto&& slot : new_reqs) {
          // initiate the task
          auto& task    = pending_[slot];
          auto& signals = signals_[rtlx::to_underlying(task.type)];
          for (auto&& sig : signals) {
            if (sig) {
              sig(task.message, mpi_reqs_[slot]);
            }
          }
        }

        new_reqs.clear();
      }

      idx_ranges_t reqs_done;

      {
        Timer{completion_time};
        reqs_done = progress_requests();
      }

      {
        Timer{callback_time};

        auto const nCompleted = std::distance(
            std::begin(indices_), reqs_done[reqs_done.size() - 1]);

        if (nCompleted) {
          trigger_callbacks(reqs_done);
        }

        progress = progress || (nCompleted > 0);
      }
    }

    stats_.iterations++;
    stats_.completion_time += completion_time;
    stats_.dispatch_time += dispatch_time;
    stats_.queue_time += queue_time;
    stats_.callback_time += callback_time;

    if (is_all_done() || progress) {
      cv_finished_.notify_all();
    }
  } while (!terminate_);
}  // namespace fmpi

template <mpi::reqsome_op testReqs>
void CommDispatcher<testReqs>::dispatch(message_type qid, Message message) {
  {
    // 1) Acquire lock
    std::lock_guard<std::mutex> lock(mutex_);

    queues_[rtlx::to_underlying(qid)].push(Task{message, qid});
  }

  // 3) signal
  cv_tasks_.notify_one();
}

template <typename mpi::reqsome_op testReqs>
inline typename CommDispatcher<testReqs>::idx_ranges_t
CommDispatcher<testReqs>::progress_requests() {
  int* last;

  auto const nreqs = req_count();
  FMPI_DBG(nreqs);

  if (!nreqs) {
    auto empty_ranges = idx_ranges_t{};
    empty_ranges.fill(std::begin(indices_));
    return empty_ranges;
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

  idx_ranges_t ranges{};
  for (auto&& type : range(n_types)) {
    auto first = (type == 0) ? std::begin(indices_) : ranges[type - 1];

    auto const limit = static_cast<int>(reqs_in_flight_ * (type + 1));

    ranges[type] = (type == n_types - 1) ? last : std::partition(
        first, last, [limit](auto const& req) { return req < limit; });

    // release request slots
    std::copy(first, ranges[type], std::front_inserter(req_slots_[type]));
  }

  return ranges;
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::trigger_callbacks(
    idx_ranges_t completed) {
  for (auto&& type : range(n_types)) {
    auto first = type == 0 ? std::begin(indices_) : completed[type - 1];
    auto last  = completed[type];

    FMPI_ASSERT(first <= last);

    // fire callbacks
    for (; first != last; ++first) {
      auto& processed = pending_[*first];

      for (auto&& cb : callbacks_[type]) {
        FMPI_ASSERT(cb);
        cb(processed.message);
      }
    }
  }
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
inline typename CommDispatcher<testReqs>::signal_token
CommDispatcher<testReqs>::register_signal(
    message_type type, F&& callable, Args&&... args) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  return signals_[slot].emplace(
      std::end(signals_[slot]),
      signal::make(
          std::forward<F>(callable), std::forward<Args...>(args)...));
}

template <mpi::reqsome_op testReqs>
template <class F, class... Args>
inline typename CommDispatcher<testReqs>::callback_token
CommDispatcher<testReqs>::register_callback(
    message_type type, F&& callable, Args&&... args) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  return callbacks_[slot].emplace(
      std::end(callbacks_[slot]),
      callback::make(
          std::forward<F>(callable), std::forward<Args...>(args)...));
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::discard_signals() {
  using callback_container = decltype(callbacks_);
  using signal_container   = decltype(signals_);

  signals_   = signal_container{};
  callbacks_ = callback_container{};
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::reset(std::size_t winsz) {
  std::lock_guard<std::mutex> lk{mutex_};

  do_reset(winsz);
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::do_reset(std::size_t winsz) {
  discard_signals();

  FMPI_ASSERT(is_all_done());

  winsz_          = winsz;
  reqs_in_flight_ = winsz_ / n_types;

  pending_.resize(winsz_);
  mpi_reqs_.resize(winsz_);
  mpi_statuses_.resize(winsz_);
  indices_.resize(winsz_);

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

  FMPI_ASSERT((winsz_ % n_types) == 0);
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
