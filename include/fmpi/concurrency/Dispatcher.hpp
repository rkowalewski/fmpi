#ifndef FMPI_CONCURRENCY_DISPATCHER_HPP
#define FMPI_CONCURRENCY_DISPATCHER_HPP

#include <mpi.h>

#include <fmpi/Message.hpp>
#include <fmpi/concurrency/BufferedChannel.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/memory/HeapAllocator.hpp>
#include <list>
#include <rtlx/Enum.hpp>
#include <tlx/container/ring_buffer.hpp>
#include <tlx/delegate.hpp>

namespace fmpi {

class CommDispatcher;

class ScheduleHandle {
  using identifier                    = int32_t;
  static constexpr identifier null_id = -2;

 public:
  constexpr ScheduleHandle() = default;
  constexpr explicit ScheduleHandle(identifier id) noexcept
    : id_(id) {
  }

  [[nodiscard]] constexpr identifier id() const noexcept {
    return id_;
  }

 private:
  identifier id_ = null_id;
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

  constexpr CommTask(ScheduleHandle id, message_type t, Message m)
    : message(m)
    , id(id)
    , type(t) {
  }

  constexpr CommTask(ScheduleHandle id, message_type t)
    : CommTask(id, t, Message{}) {
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
}

#if 0
enum class status
{
  pending,
  running,
  resolved,
  rejected
};
#endif

class ScheduleCtx {
  friend class CommDispatcher;

  using signal   = tlx::delegate<void(Message&)>;
  using callback = tlx::delegate<void(std::vector<Message>)>;

  enum class status
  {
    pending,
    error,
    ready
  };

 public:
  explicit ScheduleCtx(
      std::array<std::size_t, detail::n_types> nslots, collective_promise pr);

  void register_signal(message_type type, signal&& callable);
  void register_callback(message_type type, callback&& callable);

  // void wait();
  // bool ready() const noexcept;

 private:
  // complete all outstanding requests
  void complete_all();
  // void notify();

  /// Request Handles
  std::array<std::size_t, detail::n_types> const    nslots_;
  std::size_t const                                 winsz_;
  FixedVector<MPI_Request>                          handles_;
  FixedVector<CommTask>                             pending_;
  std::array<tlx::RingBuffer<int>, detail::n_types> slots_;

  /// Signals and Callbacks
  std::array<signal, detail::n_types>   signals_{};
  std::array<callback, detail::n_types> callbacks_{};

  /// Status Information
  status state_{status::pending};

  // promise to notify waiting tasks
  collective_promise promise_{};
};

class CommDispatcher {
  static_assert(
      detail::n_types == 2, "only four message types supported for now");

  using channel = buffered_channel<CommTask>;

  class ctx_map {
    using value_type =
        std::pair<ScheduleHandle, std::unique_ptr<ScheduleCtx>>;

    using allocator = HeapAllocator<value_type>;

    using container =
        std::list<value_type, ContiguousPoolAllocator<value_type>>;

   public:
    using iterator       = typename container::iterator;
    using const_iterator = typename container::const_iterator;

    ctx_map();
    void assign(
        ScheduleHandle const& hdl, std::unique_ptr<ScheduleCtx> /*p*/);
    void erase(iterator it);

    bool                            contains(ScheduleHandle const& hdl) const;
    std::pair<iterator, bool>       find(ScheduleHandle const& hdl);
    std::pair<const_iterator, bool> find(ScheduleHandle const& hdl) const;

    std::pair<iterator, iterator> known_schedules();

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

  ScheduleHandle submit(std::unique_ptr<ScheduleCtx> ctx);

  template <class... Args>
  void schedule(ScheduleHandle const& handle, Args&&... args) {
    FMPI_ASSERT(schedules_.contains(handle));
    auto       task   = CommTask{handle, std::forward<Args>(args)...};
    auto const status = channel_.push(task);
    FMPI_ASSERT(status == channel_op_status::success);
  }

  void commit(ScheduleHandle const& hdl);

 private:
  void        progress_all(bool blocking = false);
  static void handle_task(CommTask task, ScheduleCtx* uptr);
  void        worker();

  channel     channel_;
  ctx_map     schedules_;
  std::thread thread_;
};

CommDispatcher& static_dispatcher_pool();

}  // namespace fmpi

#endif
