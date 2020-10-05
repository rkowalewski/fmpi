#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <numeric>
#include <utility>

namespace fmpi {

namespace internal {

constexpr std::size_t      channel_capacity   = 10000;
constexpr std::size_t      schedules_capacity = 100;
static std::atomic_int32_t last_schedule_id   = 0;

}  // namespace internal

CommDispatcher& dispatcher_executor() {
  static auto dispatcher = std::make_unique<CommDispatcher>();
  return *dispatcher;
}

ScheduleCtx::ScheduleCtx(
    std::array<std::size_t, detail::n_types> nslots, collective_promise pr)
  : nslots_(nslots)
  , winsz_(std::accumulate(nslots.begin(), nslots.end(), 0u))
  , handles_(winsz_, MPI_REQUEST_NULL)
  , pending_(winsz_)
  , promise_(std::move(pr)) {
  // generate the slots
  std::size_t n = 0;
  for (auto&& i : range(detail::n_types)) {
    slots_[i] = tlx::RingBuffer<int>{nslots[i]};
    std::generate_n(
        std::front_inserter(slots_[i]), nslots[i], [&n]() { return n++; });
  }
}

inline void ScheduleCtx::complete_all() {
  FixedVector<MPI_Status> statuses(handles_.size());

  std::vector<std::size_t> idxs;
  idxs.reserve(handles_.size());

  auto r = range(handles_.size());

  std::copy_if(
      std::begin(r),
      std::end(r),
      std::back_inserter(idxs),
      [this](auto const& idx) { return handles_[idx] != MPI_REQUEST_NULL; });

  auto const ret =
      MPI_Waitall(handles_.size(), handles_.data(), statuses.data());
  FMPI_ASSERT(ret == MPI_SUCCESS);

  std::array<std::vector<Message>, detail::n_types> msgs;

  for (auto&& idx : idxs) {
    auto&      task = pending_[idx];
    auto const tid  = rtlx::to_underlying(task.type);
    FMPI_ASSERT(task.valid());

    msgs[tid].emplace_back(task.message);
  }

  for (auto&& tid : range(detail::n_types)) {
    if ((not msgs[tid].empty()) and callbacks_[tid]) {
      callbacks_[tid](msgs[tid]);
    }
  }
}  // namespace fmpi

void ScheduleCtx::register_signal(message_type type, signal&& callable) {
  FMPI_ASSERT(not signals_[rtlx::to_underlying(type)]);
  signals_[rtlx::to_underlying(type)] = std::move(callable);
}

void ScheduleCtx::register_callback(message_type type, callback&& callable) {
  FMPI_ASSERT(not callbacks_[rtlx::to_underlying(type)]);
  callbacks_[rtlx::to_underlying(type)] = std::move(callable);
}

collective_future ScheduleCtx::get_future() {
  return promise_.get_future();
}

CommDispatcher::CommDispatcher()
  : channel_(internal::channel_capacity) {
  thread_ = std::thread([this]() { worker(); });
}

CommDispatcher::~CommDispatcher() {
  channel_.close();
  thread_.join();
}

struct DefaultMessageHandler {
  int operator()(Message& message, MPI_Request& req, message_type t) const {
    if (t == message_type::IRECV) {
      return mpi::irecv(
          message.writable_buffer(),
          message.count(),
          message.type(),
          message.peer(),
          message.tag(),
          message.comm(),
          &req);
    }
    FMPI_ASSERT(t == message_type::ISEND);

    return mpi::isend(
        message.readable_buffer(),
        message.count(),
        message.type(),
        message.peer(),
        message.tag(),
        message.comm(),
        &req);
  }
};

ScheduleHandle CommDispatcher::submit(std::unique_ptr<ScheduleCtx> ctx) {
  auto hdl = ScheduleHandle{internal::last_schedule_id++};
  FMPI_ASSERT(ctx->promise_.sptr_.use_count() >= 2);
  schedules_.assign(hdl, std::move(ctx));
  return hdl;
}

void CommDispatcher::schedule(
    ScheduleHandle const& handle, message_type type, Message message) {
  // steady_timer t_enqueue{stats_.enqueue_time};

  FMPI_ASSERT(schedules_.contains(handle));
  auto const status = channel_.push(CommTask{type, message, handle});
  FMPI_ASSERT(status == channel_op_status::success);
}

void CommDispatcher::worker() {
  using namespace std::chrono_literals;
  for (;;) {
    // fetch new task, however, wait at most 1us
    CommTask task;
    auto     status = channel_.pop(task, 1us);

    if (status == channel_op_status::closed) {
      constexpr bool blocking = true;
      progress_all(blocking);
      break;
    } else if (status == channel_op_status::success) {
      FMPI_ASSERT(task.valid());

      // retrieve ctx
      auto [it, ok] = schedules_.find(task.id);
      FMPI_ASSERT(ok);

      auto& up = it->second;

      if (task.type == message_type::COMMIT) {
        FMPI_ASSERT(up->state_ == ScheduleCtx::status::pending);
        up->complete_all();
        up->promise_.set_value(MPI_SUCCESS);
        schedules_.erase(it);
        continue;
      }

      auto const ti   = rtlx::to_underlying(task.type);
      auto&      rb   = up->slots_[ti];
      int        slot = MPI_UNDEFINED;
      MPI_Status status;

      if (rb.empty()) {
        // complete one of pending requests and replace it with new slot
        std::vector<MPI_Request> reqs;
        std::vector<int>         idxs;

        auto const first_slot =
            std::accumulate(up->nslots_.begin(), up->nslots_.begin() + ti, 0);

        auto const last_slot = first_slot + up->nslots_[ti];

        FMPI_DBG(std::make_pair(first_slot, last_slot));

        for (auto&& idx : range<int>(first_slot, last_slot)) {
          if (up->pending_[idx].type == task.type) {
            reqs.emplace_back(up->handles_[idx]);
            idxs.emplace_back(idx);
          }
        }

        auto const count =
            std::count(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

        FMPI_ASSERT(reqs.empty() || count == 0);

        FMPI_DBG(reqs.size());

        int        c   = MPI_UNDEFINED;
        auto const ret = MPI_Waitany(reqs.size(), reqs.data(), &c, &status);

        FMPI_ASSERT(ret == MPI_SUCCESS);
        FMPI_ASSERT(c != MPI_UNDEFINED);

        slot               = idxs[c];
        up->handles_[slot] = MPI_REQUEST_NULL;
      } else {
        // obtain free slot from ring buffer
        slot = rb.back();
        rb.pop_back();
      }

      FMPI_ASSERT(up->handles_[slot] == MPI_REQUEST_NULL);

      // Issue new message
      if (up->signals_[ti]) {
        up->signals_[ti](task.message);
      }

      // TODO: register custom message handler for each context
      DefaultMessageHandler h{};
      auto ret = h(task.message, up->handles_[slot], task.type);

      FMPI_ASSERT(ret == MPI_SUCCESS);

      std::swap(task, up->pending_[slot]);

      // task holds now the previous task...
      // so let's complete callbacks for it

      if (task.valid() and up->callbacks_[ti]) {
        up->callbacks_[ti](std::vector<Message>({task.message}));
      }

      progress_all();
    }
  }
}

void CommDispatcher::commit(ScheduleHandle const& hdl) {
  auto status = channel_.push(CommTask{message_type::COMMIT, Message{}, hdl});
  FMPI_ASSERT(status == channel_op_status::success);
}

void CommDispatcher::progress_all(bool blocking) {
  // all requests
  std::vector<MPI_Request>  reqs;
  std::vector<ScheduleCtx*> sps;
  // map indices in reqs to tuples of (sps, req_idx)
  std::vector<std::pair<std::size_t, std::size_t>> ctx_handles;

  {
    // create a copy of all known schedules
    auto [first, last] = schedules_.known_schedules();
    sps.reserve(std::distance(first, last));

    for (auto it = first; it != last; ++it) {
      auto* up = it->second.get();
      auto& sp = sps.emplace_back(up);

      for (auto&& i : range(sp->handles_.size())) {
        if (sp->handles_[i] != MPI_REQUEST_NULL) {
          FMPI_ASSERT(sp->pending_[i].valid());
          reqs.emplace_back(sp->handles_[i]);
          ctx_handles.emplace_back(sps.size() - 1, i);
        }
      }
    }
  }

  if (reqs.empty()) {
    return;
  }

  FMPI_ASSERT(reqs.size() == ctx_handles.size());

  std::vector<int>        idxs_completed(reqs.size());
  FixedVector<MPI_Status> statuses(reqs.size());

  if (blocking) {
    auto const ret = MPI_Waitall(reqs.size(), reqs.data(), statuses.data());
    FMPI_ASSERT(ret == MPI_SUCCESS);
    std::iota(std::begin(idxs_completed), std::end(idxs_completed), 0);
  } else {
    int n = MPI_UNDEFINED;

    auto const ret = MPI_Testsome(
        reqs.size(), reqs.data(), &n, idxs_completed.data(), statuses.data());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    idxs_completed.resize((n == MPI_UNDEFINED) ? 0 : n);

    FMPI_DBG(std::make_pair(reqs.size(), idxs_completed.size()));
  }

  if (idxs_completed.empty()) {
    return;
  }

  FixedVector<std::array<std::vector<Message>, detail::n_types>> ctx_tasks(
      sps.size());

  for (auto&& i : idxs_completed) {
    // 1. mark request as complete
    auto const [sp_idx, req_idx] = ctx_handles[i];
    auto& sp                     = sps[sp_idx];

    // mark as complete
    sp->handles_[req_idx] = MPI_REQUEST_NULL;

    // 2. notify all tasks
    auto& task = sp->pending_[req_idx];

    auto const ti = rtlx::to_underlying(task.type);

    ctx_tasks[sp_idx][ti].emplace_back(task.message);

    // for (auto&& cb : sp->callbacks_[ti]) {
    //  cb(task.message);
    //}

    task.reset();

    sp->slots_[ti].push_front(req_idx);
  }

  for (auto&& sp_idx : range(ctx_tasks.size())) {
    for (auto&& ti : range(detail::n_types)) {
      // issue callbacks only if vector is non empty
      auto& msgs = ctx_tasks[sp_idx][ti];
      auto& cb   = sps[sp_idx]->callbacks_[ti];
      if (cb and not msgs.empty()) {
        cb(msgs);
      }
    }
  }
}

CommDispatcher::ctx_map::ctx_map()
  : alloc_(internal::schedules_capacity)
  , items_(alloc_) {
}

void CommDispatcher::ctx_map::assign(
    ScheduleHandle const& hdl, std::unique_ptr<ScheduleCtx> p) {
  std::lock_guard<std::mutex> lg{mtx_};
  FMPI_ASSERT(do_find(hdl) == std::end(items_));
  items_.emplace_back(hdl, std::move(p));
}

void CommDispatcher::ctx_map::erase(iterator it) {
  std::lock_guard<std::mutex> lg{mtx_};
  FMPI_ASSERT(do_find(it->first) != std::end(items_));
  items_.erase(it);
}

bool CommDispatcher::ctx_map::contains(ScheduleHandle const& hdl) const {
  std::lock_guard<std::mutex> lg{mtx_};
  return do_find(hdl) != std::end(items_);
}

std::pair<CommDispatcher::ctx_map::const_iterator, bool>
CommDispatcher::ctx_map::find(ScheduleHandle const& hdl) const {
  std::lock_guard<std::mutex> lg{mtx_};
  auto const                  it = do_find(hdl);
  return std::make_pair(it, it != items_.cend());
}

std::pair<CommDispatcher::ctx_map::iterator, bool>
CommDispatcher::ctx_map::find(ScheduleHandle const& hdl) {
  std::lock_guard<std::mutex> lg{mtx_};
  auto const                  it = do_find(hdl);
  return std::make_pair(it, it != items_.end());
}

CommDispatcher::ctx_map::const_iterator CommDispatcher::ctx_map::do_find(
    ScheduleHandle hdl) const {
  return std::find_if(items_.cbegin(), items_.cend(), [hdl](auto const& v) {
    return v.first == hdl;
  });
}

CommDispatcher::ctx_map::iterator CommDispatcher::ctx_map::do_find(
    ScheduleHandle hdl) {
  return std::find_if(items_.begin(), items_.end(), [hdl](auto const& v) {
    return v.first == hdl;
  });
}

std::
    pair<CommDispatcher::ctx_map::iterator, CommDispatcher::ctx_map::iterator>
    CommDispatcher::ctx_map::known_schedules() {
  std::lock_guard<std::mutex> lg{mtx_};
  return std::make_pair(std::begin(items_), std::end(items_));
}

}  // namespace fmpi
