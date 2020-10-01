#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/mpi/Algorithm.hpp>

std::atomic_uint32_t fmpi::ScheduleHandle::last_id_ = 0;

namespace fmpi {
ScheduleCtx::ScheduleCtx(std::array<std::size_t, detail::n_types> nslots)
  : winsz_(std::accumulate(nslots.begin(), nslots.end(), 0u))
  , handles_(winsz_, MPI_REQUEST_NULL)
  , pending_(winsz_) {
  // generate the slots
  std::size_t n = 0;
  for (auto&& i : range(detail::n_types)) {
    slots_[i] = tlx::RingBuffer<int>{nslots[i]};
    std::generate_n(
        std::front_inserter(slots_[i]), nslots[i], [&n]() { return n++; });
  }
}

void ScheduleCtx::finish() {
  std::lock_guard<std::mutex> lk{mtx_};

  auto const prev = std::exchange(state_, status::ready);
  FMPI_ASSERT(prev == status::init);
  cv_finish_.notify_all();
}

void ScheduleCtx::wait() {
  std::unique_lock<std::mutex> lk(mtx_);
  while (state_ != status::ready) {
    cv_finish_.wait(lk);
  }
}

namespace v2 {

static constexpr std::size_t channel_capacity   = 10000;
static constexpr std::size_t schedules_capacity = 100;

CommDispatcher::CommDispatcher()
  : channel_(channel_capacity) {
  thread_ = std::thread([this]() { worker(); });
}

CommDispatcher::~CommDispatcher() {
  terminate();
  thread_.join();
}

void CommDispatcher::terminate() {
  channel_.close();
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

    } else {
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
  }
};

ScheduleHandle CommDispatcher::submit(std::weak_ptr<ScheduleCtx> ctx) {
  auto hdl = ScheduleHandle{ScheduleHandle::last_id_++};
  schedules_.assign(hdl, ctx);
  return hdl;
}

void CommDispatcher::dispatch(
    ScheduleHandle handle, message_type type, Message message) {
  // steady_timer t_enqueue{stats_.enqueue_time};

  // It may be that
  // FMPI_ASSERT(schedules_.contains(handle));
  auto const status = channel_.push(CommTask{type, message, handle});
  FMPI_ASSERT(status == channel_op_status::success);
}

inline void CommDispatcher::start_worker() {
  thread_ = std::thread([this]() { worker(); });
}

void CommDispatcher::worker() {
  using namespace std::chrono_literals;
  for (;;) {
    // fetch new task, however, wait at most 1us
    CommTask task;
    auto     status = channel_.pop(task, 1us);

    if (status == channel_op_status::closed) {
      break;
    } else if (status == channel_op_status::success) {
      FMPI_ASSERT(task.valid());

      // retrieve ctx
      auto [it, ok] = schedules_.find(task.id);
      FMPI_ASSERT(ok);

      if (task.type == message_type::COMMIT) {
        if (auto sp = it->second.lock()) {
          std::vector<std::size_t> idxs;
          std::vector<MPI_Status>  statuses(sp->handles_.size());
          idxs.reserve(sp->handles_.size());

          auto r = range(sp->handles_.size());

          std::copy_if(
              std::begin(r),
              std::end(r),
              std::back_inserter(idxs),
              [p = sp.get()](auto const& idx) {
                return p->handles_[idx] != MPI_REQUEST_NULL;
              });

          auto const ret = MPI_Waitall(
              sp->handles_.size(), sp->handles_.data(), statuses.data());
          FMPI_ASSERT(ret == MPI_SUCCESS);

          for (auto&& idx : idxs) {
            auto&      task = sp->pending_[idx];
            auto const tid  = rtlx::to_underlying(task.type);
            FMPI_ASSERT(task.valid());

            for (auto&& cb : sp->callbacks_[tid]) {
              cb(task.message);
            }
          }

          sp->finish();
        }
      } else {
        if (auto ctx = it->second.lock()) {
          auto const ti   = rtlx::to_underlying(task.type);
          auto&      rb   = ctx->slots_[ti];
          int        slot = MPI_UNDEFINED;
          MPI_Status status;

          if (rb.size() == 0u) {
            // complete one of pending requests and replace it with new slot
            std::vector<MPI_Request> reqs;
            std::vector<int>         idxs;

            auto first_slot = std::accumulate(
                ctx->slots_.begin(),
                ctx->slots_.begin() + ti,
                0,
                [](auto acc, auto const& rb) -> int {
                  return acc + rb.size();
                });

            for (auto&& idx :
                 range<int>(first_slot, first_slot + rb.size())) {
              if (ctx->pending_[idx].type == task.type) {
                reqs.emplace_back(ctx->handles_[idx]);
                idxs.emplace_back(idx);
              }
            }

            auto const count = std::count(
                std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

            FMPI_ASSERT(count == 0);

            FMPI_DBG(reqs.size());

            int        c = MPI_UNDEFINED;
            auto const ret =
                MPI_Waitany(reqs.size(), reqs.data(), &c, &status);

            FMPI_ASSERT(ret == MPI_SUCCESS);
            FMPI_ASSERT(c != MPI_UNDEFINED);

            slot                = idxs[c];
            ctx->handles_[slot] = MPI_REQUEST_NULL;
          } else {
            // obtain free slot from ring buffer
            slot = rb.back();
            rb.pop_back();
          }

          FMPI_ASSERT(ctx->handles_[slot] == MPI_REQUEST_NULL);

          // Issue new message
          for (auto&& sig : ctx->signals_[ti]) {
            sig(task.message);
          }

          // TODO: register custom message handler for each context
          DefaultMessageHandler h{};
          auto ret = h(task.message, ctx->handles_[slot], task.type);

          FMPI_ASSERT(ret == MPI_SUCCESS);

          std::swap(task, ctx->pending_[slot]);

          // task holds now the previous task...
          // so let's complete callbacks for it

          if (task.valid()) {
            for (auto&& cb : ctx->callbacks_[ti]) {
              cb(task.message);
            }
          }

          // progress (test) all operations
        }
      }
    }

    progress_all();

    schedules_.release_expired();
  }
}

void CommDispatcher::commit(ScheduleHandle hdl) {
  auto status = channel_.push(CommTask{message_type::COMMIT, Message{}, hdl});
  FMPI_ASSERT(status == channel_op_status::success);
}

void CommDispatcher::progress_all(bool blocking) {
  // progress all schedules
  using schedule_t =
      typename std::iterator_traits<typename ctx_map::iterator>::value_type;

  // schedules
  std::vector<schedule_t> scheds;
  // all requests
  std::vector<MPI_Request> reqs;
  // map indices in reqs to tuples of (handle, internal idx)
  std::vector<std::pair<std::size_t, std::size_t>> ctx_handles;
  std::vector<std::shared_ptr<ScheduleCtx>>        sps;

  // create a copy of all known schedules
  schedules_.copy(std::back_inserter(scheds));
  sps.reserve(scheds.size());

  for (auto&& sched : scheds) {
    auto wp = sched.second;
    if (auto sp_ = wp.lock()) {
      auto& sp = sps.emplace_back(std::move(sp_));

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

  std::vector<int>        idxs(reqs.size());
  FixedVector<MPI_Status> statuses(reqs.size());

  if (blocking) {
    auto const ret = MPI_Waitall(reqs.size(), reqs.data(), statuses.data());
    FMPI_ASSERT(ret == MPI_SUCCESS);
    std::iota(std::begin(idxs), std::end(idxs), 0);
  } else {
    int n = MPI_UNDEFINED;

    auto const ret = MPI_Testsome(
        reqs.size(), reqs.data(), &n, idxs.data(), statuses.data());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    idxs.resize((n == MPI_UNDEFINED) ? 0 : n);
  }

  for (auto&& i : idxs) {
    // 1. mark request as complete
    auto const [sp_idx, req_idx] = ctx_handles[i];
    auto sp                      = sps[sp_idx];

    // mark as complete
    sp->handles_[req_idx] = MPI_REQUEST_NULL;

    // 2. notify all tasks
    auto& task = sp->pending_[req_idx];

    auto const ti = rtlx::to_underlying(task.type);

    for (auto&& cb : sp->callbacks_[ti]) {
      cb(task.message);
    }

    task.reset();

    sp->slots_[ti].push_front(req_idx);
  }
}

void CommDispatcher::ctx_map::release_expired() {
  std::lock_guard<std::mutex> lg{mtx_};
  items_.remove_if([](auto const& pair) { return pair.second.expired(); });
}

#if 0
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
#endif

CommDispatcher::ctx_map::ctx_map()
  : alloc_(schedules_capacity)
  , items_(alloc_) {
}

void CommDispatcher::ctx_map::assign(
    ScheduleHandle hdl, std::weak_ptr<ScheduleCtx> p) {
  std::lock_guard<std::mutex> lg{mtx_};
  FMPI_ASSERT(do_find(hdl) == std::end(items_));
  items_.emplace_back(hdl, p);
}

void CommDispatcher::ctx_map::merge(CommDispatcher::ctx_map::container&&) {
}

void CommDispatcher::ctx_map::erase(CommDispatcher::ctx_map::iterator it) {
  std::lock_guard<std::mutex> lg{mtx_};
  FMPI_ASSERT(it != std::end(items_));
  items_.erase(it);
}

bool CommDispatcher::ctx_map::contains(ScheduleHandle hdl) const {
  std::lock_guard<std::mutex> lg{mtx_};
  return do_find(hdl) != std::end(items_);
}

std::pair<CommDispatcher::ctx_map::const_iterator, bool>
CommDispatcher::ctx_map::find(ScheduleHandle hdl) const {
  std::lock_guard<std::mutex> lg{mtx_};
  auto const                  it = do_find(hdl);
  return std::make_pair(it, it != items_.cend());
}

std::pair<CommDispatcher::ctx_map::iterator, bool>
CommDispatcher::ctx_map::find(ScheduleHandle hdl) {
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

}  // namespace v2
}  // namespace fmpi
