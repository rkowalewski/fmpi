#ifndef FMPI_CONCURRENCY_DISPATCHER_HPP
#define FMPI_CONCURRENCY_DISPATCHER_HPP

#include <mpi.h>

#include <atomic>
#include <fmpi/Debug.hpp>
#include <fmpi/Function.hpp>
#include <fmpi/Message.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/concurrency/BufferedChannel.hpp>
#include <fmpi/concurrency/SPSC.hpp>
#include <fmpi/concurrency/SimpleConcurrentDeque.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/memory/HeapAllocator.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/mpi/Request.hpp>
#include <fmpi/util/Trace.hpp>
#include <gsl/span>
#include <list>
#include <mutex>
#include <new>
#include <numeric>
#include <queue>
#include <rtlx/Enum.hpp>
#include <rtlx/Timer.hpp>
#include <thread>
#include <tlx/container/ring_buffer.hpp>
#include <tlx/container/simple_vector.hpp>
#include <tlx/delegate.hpp>
#include <unordered_map>

namespace fmpi {

namespace v2 {
class CommDispatcher;

}

class ScheduleHandle {
  using identifier = uint32_t;

 public:
  constexpr ScheduleHandle() = default;
  constexpr explicit ScheduleHandle(identifier id) noexcept
    : id_(id) {
  }

  [[nodiscard]] constexpr identifier id() const noexcept {
    return id_;
  }

 private:
  identifier id_ = -1;
};

constexpr bool operator==(
    ScheduleHandle const& lhs, ScheduleHandle const& rhs) {
  return lhs.id() == rhs.id();
}

constexpr bool operator!=(
    ScheduleHandle const& lhs, ScheduleHandle const& rhs) {
  return not(lhs.id() == rhs.id());
}

struct CommTask {
  constexpr CommTask() = default;

  constexpr CommTask(
      message_type t, Message m, ScheduleHandle id = ScheduleHandle{})
    : message(m)
    , id(id)
    , type(t) {
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return type != message_type::INVALID;
  }

  constexpr void reset() noexcept {
    type = message_type::INVALID;
  }

  Message        message{};
  ScheduleHandle id{};
  message_type   type{message_type::INVALID};
};

namespace detail {
static constexpr auto n_types = rtlx::to_underlying(message_type::INVALID);

int dispatch_waitall(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last);

template <class T>
struct ContiguousList {
  using allocator = HeapAllocator<T>;
  using container = std::list<T, ContiguousPoolAllocator<T>>;
};

class MultiTaskQueue {
  using value_type = CommTask;
  using container  = ContiguousList<value_type>;
  using allocator  = typename container::allocator;
  using list       = typename container::container;

  using task_queue = std::queue<value_type, list>;

  using reference       = typename task_queue::reference;
  using const_reference = typename task_queue::const_reference;

  static constexpr uint16_t default_cap = 1000;

 public:
  explicit MultiTaskQueue(uint16_t initial_cap = default_cap)
    : alloc_(initial_cap) {
    for (auto&& i : range(lists_.size())) {
      lists_[i] = task_queue{alloc_};
    }
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return std::accumulate(
        std::begin(lists_),
        std::end(lists_),
        0U,
        [](auto acc, auto const& queue) { return acc + queue.size(); });
  }

  [[nodiscard]] std::size_t size(message_type type) const noexcept {
    return lists_[rtlx::to_underlying(type)].size();
  }

  void push(value_type const& task) {
    lists_[rtlx::to_underlying(task.type)].push(task);
  }

  void push(value_type&& task) {
    lists_[rtlx::to_underlying(task.type)].emplace(task);
  }

  void pop(message_type type) {
    return lists_[rtlx::to_underlying(type)].pop();
  }

  reference front(message_type type) {
    return lists_[rtlx::to_underlying(type)].front();
  }

  [[nodiscard]] const_reference front(message_type type) const {
    return lists_[rtlx::to_underlying(type)].front();
  }

 private:
  allocator alloc_;

  // Task Lists: One for each type
  std::array<task_queue, n_types> lists_{};
};

}  // namespace detail

enum class status
{
  pending,
  running,
  resolved,
  rejected
};

class ScheduleCtx {
  friend class v2::CommDispatcher;

  using signal   = tlx::delegate<void(Message&)>;
  using callback = tlx::delegate<void(std::vector<Message>)>;

  enum class status
  {
    pending,
    ready
  };

 public:
  explicit ScheduleCtx(std::array<std::size_t, detail::n_types> nslots);

  void register_signal(message_type type, signal&& callable);
  void register_callback(message_type type, callback&& callable);

  void wait();
  bool ready() const noexcept;

 private:
  // complete all outstanding requests
  void complete_all();
  void notify();

  /// Request Handles
  std::array<std::size_t, detail::n_types> const    nslots_;
  std::size_t const                                 winsz_;
  FixedVector<MPI_Request>                          handles_;
  FixedVector<CommTask>                             pending_;
  std::array<tlx::RingBuffer<int>, detail::n_types> slots_;

  /// Signals and Callbacks
  // std::array<std::list<signal>, detail::n_types>   signals_;
  // std::array<std::list<callback>, detail::n_types> callbacks_;
  std::array<signal, detail::n_types>   signals_{};
  std::array<callback, detail::n_types> callbacks_{};

  /// Status Information
  std::mutex              mtx_;
  std::condition_variable cv_finish_;
  std::atomic<status>     state_{status::pending};
};

namespace v2 {
class CommDispatcher {
  static_assert(
      detail::n_types == 2, "only four message types supported for now");

  using channel = buffered_channel<CommTask>;

  class ctx_map {
    using value_type = std::pair<ScheduleHandle, std::weak_ptr<ScheduleCtx>>;
    using allocator  = HeapAllocator<value_type>;

    using container =
        std::list<value_type, ContiguousPoolAllocator<value_type>>;

   public:
    using iterator       = typename container::iterator;
    using const_iterator = typename container::const_iterator;

    ctx_map();
    void assign(
        ScheduleHandle const& hdl, const std::weak_ptr<ScheduleCtx>& wp);

    bool                            contains(ScheduleHandle const& hdl) const;
    std::pair<iterator, bool>       find(ScheduleHandle const& hdl);
    std::pair<const_iterator, bool> find(ScheduleHandle const& hdl) const;

    template <class OutputIterator>
    void copy(OutputIterator d_first);

    void release_expired();

   private:
    iterator       do_find(ScheduleHandle hdl);
    const_iterator do_find(ScheduleHandle hdl) const;

    mutable std::mutex mtx_;
    allocator          alloc_;
    container          items_;
  };

 public:
  CommDispatcher();
  ~CommDispatcher();

  CommDispatcher(CommDispatcher const&) = delete;
  CommDispatcher& operator=(CommDispatcher const&) = delete;

  ScheduleHandle submit(const std::weak_ptr<ScheduleCtx>& ctx);

  void schedule(
      ScheduleHandle const& handle, message_type type, Message message);

  void commit(ScheduleHandle const& hdl);

 private:
  void progress_all(bool blocking = false);
  void worker();

  channel     channel_;
  ctx_map     schedules_;
  std::thread thread_;
};

template <class OutputIterator>
void CommDispatcher::ctx_map::copy(OutputIterator d_first) {
  std::lock_guard<std::mutex> lg{mtx_};
  std::copy(items_.begin(), items_.end(), d_first);
}

}  // namespace v2

v2::CommDispatcher& dispatcher_executor();

namespace detail {

class future_shared_state {
  using mpi_result           = int;
  using simple_message_queue = SimpleConcurrentDeque<Message>;

  // the schedule context
  std::shared_ptr<ScheduleCtx> ctx_;
  // the queue to notify about ready tasks
  std::shared_ptr<simple_message_queue> q_;

 public:
  future_shared_state() noexcept = default;

  future_shared_state(
      std::shared_ptr<ScheduleCtx>          sp,
      std::shared_ptr<simple_message_queue> q);

  void       wait();
  bool       is_ready() const noexcept;
  bool       valid() const noexcept;
  mpi_result get();
};

}  // namespace detail

class when_any_executor;

class collective_future : private detail::future_shared_state {
  using base = detail::future_shared_state;

  friend class when_any_executor;

 public:
  // Constructors
  collective_future() noexcept = default;

  collective_future(
      std::shared_ptr<ScheduleCtx>                    sp,
      std::shared_ptr<SimpleConcurrentDeque<Message>> q);

  ~collective_future();

  // Only Movable
  collective_future(const collective_future&) = delete;
  collective_future(collective_future&&)      = default;

  collective_future& operator=(const collective_future&) = delete;
  collective_future& operator=(collective_future&&) = default;

  using base::get;
  using base::is_ready;
  using base::valid;
  using base::wait;
};

class when_any_executor {
  auto when_any(collective_future& fut);
  auto when_some(collective_future& fut);
};

#if 0
template <mpi::reqsome_op testReqs = mpi::testsome>
class CommDispatcher {
  static_assert(
      detail::n_types == 2, "only four message types supported for now");
  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using signal        = tlx::delegate<int(Message&, MPI_Request&)>;
  using callback      = tlx::delegate<void(Message&)>;
  using signal_list   = std::list<signal>;
  using callback_list = std::list<callback>;

  using signal_token   = typename signal_list::const_iterator;
  using callback_token = typename callback_list::const_iterator;

  using idx_ranges_t = std::array<int*, detail::n_types>;

  static constexpr auto high_watermark =
      std::string_view("Tcomm.high_watermark");
  static constexpr auto iterations = kCommRounds;
  static constexpr auto nreqs_completion =
      std::string_view("Tcomm.nreqs_completion");
  static constexpr auto dispatch_time =
      std::string_view("Tcomm.dispatch_time");
  static constexpr auto callback_time =
      std::string_view("Tcomm.callback_time");
  static constexpr auto progress_time =
      std::string_view("Tcomm.progress_time");
  static constexpr auto total_time = std::string_view("Tcomm.total_time");

 public:
  using channel = SPSCNChannel<CommTask>;

 private:
  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  std::shared_ptr<channel> task_channel_{};

  MultiTrace stats_{};

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
  simple_vector<int> indices_{};
  // free slots for each request type
  simple_vector<tlx::RingBuffer<int>> req_slots_{detail::n_types};

  // Task Cache
  detail::MultiTaskQueue backlog_{};

  // Signals
  std::array<signal_list, detail::n_types>   signals_{};
  std::array<callback_list, detail::n_types> callbacks_{};

  std::thread thread_;

 public:
  explicit CommDispatcher(std::shared_ptr<channel> chan, std::size_t winsz);

  ~CommDispatcher();

  template <class F>
  signal_token register_signal(message_type type, F&& callable);

  template <class F>
  callback_token register_callback(message_type type, F&& callable);

  void loop_until_done();

  void stop_worker();

  void start_worker();

  void reset(std::size_t winsz);

  void pinToCore(int coreId);

  typename MultiTrace::cache const& stats() const noexcept;

  /// New Interface here

 private:
  std::size_t req_count() const noexcept;

  std::size_t req_capacity() const noexcept;

  [[nodiscard]] bool has_active_requests() const noexcept;

  void worker();

  void do_dispatch(CommTask task);

  void do_progress(bool force = false);

  idx_ranges_t progress_network(bool force = false);

  void discard_signals();

  void do_init(std::size_t winsz);

  void trigger_callbacks(idx_ranges_t completed);
};

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(
    std::shared_ptr<channel> chan, std::size_t winsz)
  : task_channel_(std::move(chan)) {
  FMPI_DBG_STREAM("CommDispatcher(winsz)" << winsz);
  do_init(winsz);
}

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::~CommDispatcher() {
  stop_worker();
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::start_worker() {
  thread_ = std::thread([this]() { worker(); });
}

template <typename mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::worker() {
  constexpr auto sleep_interval = std::chrono::microseconds(1);

  struct HasTasks {
    std::pair<std::size_t, bool> task_count;

    constexpr HasTasks() = default;

    constexpr HasTasks(std::size_t first, bool second) noexcept
      : task_count(std::make_pair(first, second)) {
    }

    constexpr explicit operator bool() const noexcept {
      return (task_count.first != 0u) || task_count.second;
    }
  };

  steady_timer t_total{stats_.duration(total_time)};

  while (auto const tasks =
             HasTasks{backlog_.size(), !task_channel_->done()}) {
    if (auto const n_backlog = tasks.task_count.first; n_backlog != 0u) {
      // try to serve from caches
      for (auto&& type : range(detail::n_types)) {
        auto const nslots   = req_slots_[type].size();
        auto const req_type = static_cast<message_type>(type);
        auto const ntasks   = backlog_.size(req_type);
        for (auto&& unused : range(std::min(nslots, ntasks))) {
          std::ignore = unused;
          do_dispatch(backlog_.front(req_type));
          backlog_.pop(req_type);
        }
      }
      auto& hw = stats_.value<int64_t>(high_watermark);
      hw       = std::max<int64_t>(hw, n_backlog);
    }
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

      if (nslots == 0u) {
        backlog_.push(task);
      } else {
        do_dispatch(task);
      }
    }

    do_progress();
  }

  {
    constexpr auto force_progress           = true;
    stats_.value<int64_t>(nreqs_completion) = req_count();
    do_progress(force_progress);
  }

  {
    std::lock_guard<std::mutex> lk{mutex_};

    auto const was_busy = std::exchange(busy_, false);
    FMPI_ASSERT(was_busy);
  }

  cv_finished_.notify_all();
}

template <typename mpi::reqsome_op testReqs>
void CommDispatcher<testReqs>::do_dispatch(CommTask task) {
  steady_timer t_dispatch{stats_.duration(dispatch_time)};

  auto const req_type = rtlx::to_underlying(task.type);

  FMPI_ASSERT(req_slots_[req_type].size());

  // get a free slot
  auto const slot = req_slots_[req_type].back();
  req_slots_[req_type].pop_back();

  for (auto&& sig : signals_[req_type]) {
    sig(task.message, mpi_reqs_[slot]);
  }

  pending_[slot] = task;
}

template <typename mpi::reqsome_op testReqs>
void CommDispatcher<testReqs>::do_progress(bool force) {
  auto const reqs_done = progress_network(force);

  auto const nCompleted =
      std::distance(std::begin(indices_), reqs_done[reqs_done.size() - 1]);

  if (nCompleted) {
    trigger_callbacks(reqs_done);
  }
}

template <typename mpi::reqsome_op testReqs>
inline typename CommDispatcher<testReqs>::idx_ranges_t
CommDispatcher<testReqs>::progress_network(bool force) {
  steady_timer t_progress{stats_.duration(progress_time)};

  auto const nreqs = req_count();

  if (!nreqs) {
    auto empty_ranges = idx_ranges_t{};
    empty_ranges.fill(std::begin(indices_));
    return empty_ranges;
  }

  stats_.value<int64_t>(iterations)++;

  auto op = force ? detail::dispatch_waitall : testReqs;

  int* last = nullptr;
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
  for (auto&& type : range(detail::n_types)) {
    auto first = (type == 0) ? std::begin(indices_) : ranges[type - 1];

    auto const pivot = static_cast<int>(reqs_in_flight_ * (type + 1));

    auto mid = (type == detail::n_types - 1)
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
  steady_timer t_callback{stats_.duration(callback_time)};
  for (auto&& type : range(detail::n_types)) {
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
  FMPI_ASSERT(pin_success);
}

template <mpi::reqsome_op testReqs>
typename MultiTrace::cache const& CommDispatcher<testReqs>::stats()
    const noexcept {
  return stats_.values();
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::loop_until_done() {
  std::unique_lock<std::mutex> lk{mutex_};

  if (busy_) {
    cv_finished_.wait(lk, [this]() { return !busy_; });
  }
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::stop_worker() {
  loop_until_done();

  if (thread_.joinable()) {
    thread_.join();
  }
}

template <mpi::reqsome_op testReqs>
template <class F>
inline typename CommDispatcher<testReqs>::signal_token
CommDispatcher<testReqs>::register_signal(message_type type, F&& callable) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  return signals_[slot].emplace(
      std::end(signals_[slot]), signal::make(std::forward<F>(callable)));
}

template <mpi::reqsome_op testReqs>
template <class F>
inline typename CommDispatcher<testReqs>::callback_token
CommDispatcher<testReqs>::register_callback(message_type type, F&& callable) {
  auto const slot = rtlx::to_underlying(type);

  std::lock_guard<std::mutex> lg{mutex_};

  return callbacks_[slot].emplace(
      std::end(callbacks_[slot]), callback::make(std::forward<F>(callable)));
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::discard_signals() {
  using callback_container = std::array<callback_list, detail::n_types>;
  using signal_container   = std::array<signal_list, detail::n_types>;

  signals_   = signal_container{};
  callbacks_ = callback_container{};
}

template <mpi::reqsome_op testReqs>
inline void CommDispatcher<testReqs>::do_init(std::size_t winsz) {
  discard_signals();

  winsz_          = winsz;
  reqs_in_flight_ = winsz_ / detail::n_types;

  pending_.resize(winsz_);
  mpi_reqs_.resize(winsz_);
  mpi_statuses_.resize(winsz_);
  indices_.resize(winsz_);

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

  FMPI_ASSERT((winsz_ % detail::n_types) == 0);
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

namespace detail {
inline int dispatch_waitall(
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
#endif

}  // namespace fmpi

#endif
