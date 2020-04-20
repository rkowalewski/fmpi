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
#include <fmpi/container/IntrusiveList.hpp>
#include <fmpi/container/buffered_channel.hpp>
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

struct CommTask {
  CommTask() = default;

  CommTask(message_type t, Message m)
    : message(m)
    , type(t) {
  }

  Message      message{};
  message_type type{};
};

namespace detail {

int dispatch_waitall(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last) {
  auto const n = std::distance(begin, end);
  last         = indices;

  std::vector<int> pending_indices;
  pending_indices.reserve(n);

  for (auto&& idx : range(n)) {
    if (*std::next(begin, idx) != MPI_REQUEST_NULL) {
      pending_indices.emplace_back(idx);
    }
  }

  auto ret = MPI_Waitall(n, begin, statuses);

  if (ret == MPI_SUCCESS) {
    last = std::copy(
        std::begin(pending_indices), std::end(pending_indices), indices);
  }

  return ret;
}

}  // namespace detail

class CommChannel {
  using timer = rtlx::Timer<>;

  using duration = std::chrono::microseconds;

  using timer_duration = typename timer::duration;

 public:
  using value_type = CommTask;
  using channel    = buffered_channel<value_type>;

  CommChannel() = default;

  CommChannel(std::size_t n) noexcept
    : channel_(n)
    , count_(n) {
    FMPI_DBG("< CommChannel(chan, n)");
    FMPI_DBG(task_count());
  }

  CommChannel(CommChannel const&) = delete;
  CommChannel& operator=(CommChannel const&) = delete;

  bool wait_dequeue(value_type& val) {
    timer{time_};
    if (done()) {
      return false;
    }
    val = channel_.value_pop();
    count_ -= 1;
    return true;
  }

  bool wait_dequeue(value_type& val, duration const& timeout) {
    timer{time_};
    if (done()) {
      return false;
    }
    auto const ret = channel_.pop(val, timeout);
    count_ -= ret;
    return ret;
  }

  bool enqueue(CommTask task) {
    auto status = channel_.push(task);
    auto ret    = status == channel_op_status::success;
    FMPI_ASSERT(ret);
    return ret;
  }

  void close() {
    channel_.close();
  }

  bool done() const noexcept {
    return task_count() == 0u;
  }

  std::size_t task_count() const noexcept {
    return count_.load(std::memory_order_acquire);
    // return count_;
  }

  timer_duration time() const noexcept {
    return time_;
  }

 private:
  channel        channel_{0};
  timer_duration time_{};
  // prevent false sharing because producer may check count while the consumer
  // only changes time
  alignas(std::hardware_destructive_interference_size)
      std::atomic<std::size_t> count_{0};
};

template <mpi::reqsome_op testReqs = mpi::testsome>
class CommDispatcher {
  using Timer    = rtlx::Timer<>;
  using duration = typename Timer::duration;

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

  using contiguous_heap_allocator = HeapAllocator<CommTask, false>;
  using task_list =
      std::list<CommTask, ContiguousPoolAllocator<CommTask, false>>;
  using task_queue = std::queue<CommTask, task_list>;

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

  std::shared_ptr<CommChannel> task_channel_{};

  // Task Lists: One for each type
  std::array<task_queue, n_types> backlog_{};

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

  // flag if dispatcher is busy
  bool busy_{true};

  // flag to force termination
  bool terminate_{false};

  //////////////////////////////////
  // Sliding Window Worker Thread //
  //////////////////////////////////

  // Window Size
  std::size_t winsz_{};
  // Requests in flight for each type, currently: winsz / n_types
  std::size_t reqs_in_flight_{};
  // Pending tasks: explicitly use a normal vector to default construct
  tlx::SimpleVector<CommTask> pending_{};
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
  explicit CommDispatcher(
      std::shared_ptr<CommChannel> chan, std::size_t winsz);

  ~CommDispatcher();

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

  idx_ranges_t progress_requests(bool force = false);

  void worker();

  void do_dispatch(CommTask task);

  void discard_signals();

  void do_reset(std::size_t winsz);

  void trigger_callbacks(idx_ranges_t ranges);
};  // namespace fmpi

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(
    std::shared_ptr<CommChannel> chan, std::size_t winsz)
  : task_channel_(std::move(chan))
  , alloc_(contiguous_heap_allocator{default_task_capacity}) {
  do_reset(winsz);
}

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::~CommDispatcher() {
  loop_until_done();
  thread_.join();
}

#if 0
template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::stop_worker() {
  {
    std::lock_guard<std::mutex> lk{mutex_};
    terminate_ = true;
  }

  cv_tasks_.notify_all();
}
#endif

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::start_worker() {
  thread_ = std::thread([this]() { worker(); });
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::worker() {
  constexpr auto sleep_interval = std::chrono::microseconds(1);

  while (queue_count() || !task_channel_->done()) {
    // try to serve from caches
    for (auto&& req_type : range(n_types)) {
      auto const nslots = req_slots_[req_type].size();
      auto const ntasks = backlog_[req_type].size();
      for (auto&& unused : range(std::min(nslots, ntasks))) {
        std::ignore = unused;
        do_dispatch(backlog_[req_type].front());
        backlog_[req_type].pop();
      }
    }

    FMPI_DBG(queue_count());
    FMPI_DBG(task_channel_->task_count());
    FMPI_DBG(req_capacity());

    while (req_capacity()) {
      CommTask task{};
      // if we fail, there are two reasons...
      // either the timeout is hit without popping an element
      // or we already popped everything out of the channel
      if (!task_channel_->wait_dequeue(task, sleep_interval)) {
        break;
      }

      auto const req_type = rtlx::to_underlying(task.type);
      auto const nslots   = req_slots_[req_type].size();

      if (!nslots) {
        backlog_[req_type].push(std::move(task));
      } else {
        do_dispatch(std::move(task));
      }
    }

    auto const reqs_done = progress_requests();

    auto const nCompleted =
        std::distance(std::begin(indices_), reqs_done[reqs_done.size() - 1]);

    if (nCompleted) {
      trigger_callbacks(reqs_done);
    }
  }

  constexpr auto force     = true;
  auto const     reqs_done = progress_requests(force);

  auto const nCompleted =
      std::distance(std::begin(indices_), reqs_done[reqs_done.size() - 1]);

  if (nCompleted) {
    trigger_callbacks(reqs_done);
  }

#if 0
  stats_.iterations++;
  stats_.completion_time += completion_time;
  stats_.dispatch_time += dispatch_time;
  stats_.queue_time += queue_time;
  stats_.callback_time += callback_time;
#endif

  {
    std::lock_guard<std::mutex> lk{mutex_};

    auto const was_busy = std::exchange(busy_, false);
    FMPI_ASSERT(was_busy);
  }

  cv_finished_.notify_all();
}

template <typename mpi::reqsome_op testReqs>
void CommDispatcher<testReqs>::do_dispatch(CommTask task) {
  auto const req_type = rtlx::to_underlying(task.type);

  FMPI_ASSERT(req_slots_[req_type].size());

  // get a free slot
  auto const slot = req_slots_[req_type].back();
  req_slots_[req_type].pop_back();

  for (auto&& sig : signals_[req_type]) {
    sig(task.message, mpi_reqs_[slot]);
  }

  pending_[slot] = std::move(task);
}

template <typename mpi::reqsome_op testReqs>
inline typename CommDispatcher<testReqs>::idx_ranges_t
CommDispatcher<testReqs>::progress_requests(bool force) {
  int* last;

  auto const nreqs = req_count();
  FMPI_DBG(nreqs);
  FMPI_DBG(force);

  if (!nreqs) {
    auto empty_ranges = idx_ranges_t{};
    empty_ranges.fill(std::begin(indices_));
    return empty_ranges;
  }

  auto op = force ? detail::dispatch_waitall : testReqs;

  auto mpi_ret =
      op(&*std::begin(mpi_reqs_),
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

    auto const pivot = static_cast<int>(reqs_in_flight_ * (type + 1));

    auto mid = (type == n_types - 1)
                   ? last
                   : std::partition(first, last, [pivot](auto const& req) {
                       return req < pivot;
                     });

    ranges[type] = mid;

    // release request slots
    std::copy(first, mid, std::front_inserter(req_slots_[type]));
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

  cv_finished_.wait(lk, [this]() { return task_channel_->done() && !busy_; });
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
  using callback_container = std::array<callback_list, n_types>;
  using signal_container   = std::array<signal_list, n_types>;

  signals_   = signal_container{};
  callbacks_ = callback_container{};
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::do_reset(std::size_t winsz) {
  discard_signals();

  winsz_          = winsz;
  reqs_in_flight_ = winsz_ / n_types;

  pending_.resize(winsz_);
  mpi_reqs_.resize(winsz_);
  mpi_statuses_.resize(winsz_);
  indices_.resize(winsz_);

  for (auto&& i : range(backlog_.size())) {
    backlog_[i] = std::move(task_queue{alloc_});
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
      std::begin(backlog_),
      std::end(backlog_),
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

}  // namespace fmpi

#endif
