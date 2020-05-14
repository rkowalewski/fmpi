#ifndef FMPI_DISPATCHER_HPP
#define FMPI_DISPATCHER_HPP

#include <mpi.h>

#include <atomic>
#include <list>
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

#include <fmpi/Debug.hpp>
#include <fmpi/Function.hpp>
#include <fmpi/Message.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/memory/HeapAllocator.hpp>
#include <fmpi/mpi/Request.hpp>
#include <fmpi/util/Trace.hpp>

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

  INVALID
};

struct CommTask {
  constexpr CommTask() = default;

  constexpr CommTask(message_type t, Message m)
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

template <class T>
class SPSCNChannel {
  using duration = typename steady_timer::duration;

 public:
  struct Stats {
    // consumer
    alignas(kCacheAlignment) duration dequeue_time{0};
    // producer
    alignas(kCacheAlignment) duration enqueue_time{0};
  };

 public:
  using value_type = T;
  using channel    = buffered_channel<value_type>;

  SPSCNChannel() = default;

  SPSCNChannel(std::size_t n) noexcept
    : channel_(n)
    , high_watermark_(n)
    , count_(n) {
    FMPI_DBG("< SPSCNChannel(chan, n)");
    FMPI_DBG(task_count());
  }

  SPSCNChannel(SPSCNChannel const&) = delete;
  SPSCNChannel& operator=(SPSCNChannel const&) = delete;

  bool wait_dequeue(value_type& val) {
    steady_timer t_dequeue{stats_.dequeue_time};
    if (!count_.load(std::memory_order_relaxed)) {
      return false;
    }
    val = channel_.value_pop();
    count_.fetch_sub(1, std::memory_order_release);
    return true;
  }

  template <class Rep, class Period>
  bool wait_dequeue(
      value_type& val, std::chrono::duration<Rep, Period> const& timeout) {
    steady_timer t_enqueue{stats_.dequeue_time};
    if (!count_.load(std::memory_order_relaxed)) {
      return false;
    }
    auto const ret = channel_.pop(val, timeout);
    count_.fetch_sub(ret, std::memory_order_release);
    return ret;
  }

  bool enqueue(value_type const& task) {
    steady_timer t_enqueue{stats_.enqueue_time};
    auto const   status = channel_.push(task);
    auto const   ret    = status == channel_op_status::success;

    if ((nproduced_ += ret) == high_watermark_) {
      // channel_.close();
    }

    return ret;
  }

  [[nodiscard]] bool done() const noexcept {
    return task_count() == 0u;
  }

  [[nodiscard]] std::size_t task_count() const noexcept {
    return count_.load(std::memory_order_acquire);
  }

  Stats statistics() const noexcept {
    FMPI_ASSERT(done());
    return stats_;
  }

  [[nodiscard]] std::size_t high_watermark() const noexcept {
    return high_watermark_;
  }

 private:
  channel channel_{0};

  // two separate write cachelines
  // L1: consumer
  // L2: producer
  Stats stats_{};

  // private access in producer
  std::size_t nproduced_{0};

  // shared read cacheline
  alignas(kCacheAlignment) std::size_t const high_watermark_{0};

  // shared write cacheline
  alignas(kCacheAlignment) std::atomic<std::size_t> count_{0};
};

template <mpi::reqsome_op testReqs = mpi::testsome>
class CommDispatcher {
  static constexpr auto n_types = rtlx::to_underlying(message_type::INVALID);
  static_assert(n_types == 2, "only two request types supported for now");

  class TaskQueue;

  template <class T>
  struct ContiguousList {
    using allocator = HeapAllocator<T, false>;
    using container = std::list<T, ContiguousPoolAllocator<T, false>>;
  };

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using signal        = Function<int(Message&, MPI_Request&)>;
  using callback      = Function<void(Message&)>;
  using signal_list   = std::list<signal>;
  using callback_list = std::list<callback>;

  using signal_token   = typename signal_list::const_iterator;
  using callback_token = typename callback_list::const_iterator;

  using req_idx_t    = simple_vector<int>;
  using idx_ranges_t = std::array<int*, n_types>;

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
  //static constexpr auto completion_time =
  //    std::string_view("Tcomm.completion_time");
  static constexpr auto total_time = std::string_view("Tcomm.total_time");

 public:
  using channel = SPSCNChannel<CommTask>;

 private:
  //////////////////////////////
  // Thread Safe Work Sharing //
  //////////////////////////////

  std::shared_ptr<channel> task_channel_{};

  std::array<signal_list, n_types>   signals_{};
  std::array<callback_list, n_types> callbacks_{};

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

  // Task Cache
  TaskQueue backlog_{};

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
  explicit CommDispatcher(std::shared_ptr<channel> chan, std::size_t winsz);

  ~CommDispatcher();

  template <class F, class... Args>
  signal_token register_signal(
      message_type type, F&& callable, Args&&... args);

  template <class F, class... Args>
  callback_token register_callback(
      message_type type, F&& callable, Args&&... args);

  void loop_until_done();

  void stop_worker();

  void start_worker();

  void reset(std::size_t winsz);

  void pinToCore(int coreId);

  typename MultiTrace::cache const& stats() const noexcept;

 private:
  std::size_t req_count() const noexcept;

  std::size_t req_capacity() const noexcept;

  [[nodiscard]] bool has_active_requests() const noexcept;

  void worker();

  void do_dispatch(CommTask task);

  void do_progress(bool force = false);

  idx_ranges_t progress_network(bool force = false);

  void discard_signals();

  void do_reset(std::size_t winsz);

  void trigger_callbacks(idx_ranges_t ranges);

  class TaskQueue {
    using value_type      = CommTask;
    using contiguous_list = ContiguousList<value_type>;
    using allocator       = typename contiguous_list::allocator;
    using list            = typename contiguous_list::container;

    using task_queue = std::queue<value_type, list>;

    using reference       = typename task_queue::reference;
    using const_reference = typename task_queue::const_reference;

    static constexpr uint16_t default_task_capacity = 1000;

   public:
    TaskQueue() {
      for (auto&& i : range(lists_.size())) {
        lists_[i] = std::move(task_queue{alloc_});
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
      return lists_[rtlx::to_underlying(task.type)].push(task);
    }

    void push(value_type&& task) {
      return lists_[rtlx::to_underlying(task.type)].emplace(task);
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
    allocator alloc_{default_task_capacity};

    // Task Lists: One for each type
    std::array<task_queue, n_types> lists_{};
  };
};  // namespace fmpi

template <typename mpi::reqsome_op testReqs>
inline CommDispatcher<testReqs>::CommDispatcher(
    std::shared_ptr<channel> chan, std::size_t winsz)
  : task_channel_(std::move(chan)) {
  FMPI_DBG_STREAM("CommDispatcher(winsz)" << winsz);
  do_reset(winsz);
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

    constexpr operator bool() const noexcept {
      return task_count.first || task_count.second;
    }
  };

  steady_timer t_total{stats_.duration(total_time)};

  while (auto const tasks =
             HasTasks{backlog_.size(), !task_channel_->done()}) {
    auto n_backlog = tasks.task_count.first;
    if (n_backlog) {
      // try to serve from caches
      for (auto&& type : range(n_types)) {
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

      if (!nslots) {
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

  int* last;
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
  steady_timer t_callback{stats_.duration(callback_time)};
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
  FMPI_ASSERT(pin_success);
}

template <mpi::reqsome_op testReqs>
typename MultiTrace::cache const&
CommDispatcher<testReqs>::stats() const noexcept {
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
