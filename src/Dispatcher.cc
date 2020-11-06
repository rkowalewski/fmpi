#include <fmpi/Debug.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>
#include <numeric>
#include <utility>

namespace fmpi {

namespace internal {

constexpr std::size_t      channel_capacity   = 10000;
constexpr std::size_t      schedules_capacity = 100;
static std::atomic_int32_t last_schedule_id   = 0;

static constexpr auto t_complete_all  = std::string_view("t_complete_all");
static constexpr auto t_test_all      = std::string_view("t_test_all");
static constexpr auto t_complete_any  = std::string_view("t_complete_any");
static constexpr auto t_complete_some = std::string_view("t_complete_some");
static constexpr auto t_dispatch      = std::string_view("t_dispatch");
static constexpr auto t_copy          = std::string_view("t_copy");
static constexpr auto trace_name      = std::string_view("schedule_ctx");

}  // namespace internal

CommDispatcher& static_dispatcher_pool() {
  static auto dispatcher = std::make_unique<CommDispatcher>();
  return *dispatcher;
}

ScheduleCtx::ScheduleCtx(
    std::array<std::size_t, detail::n_types> nslots,
    collective_promise                       pr,
    std::size_t                              max_tasks)
  : nslots_(nslots)
  , winsz_(std::accumulate(nslots.begin(), nslots.end(), 0u))
  , handles_(winsz_, MPI_REQUEST_NULL)
  , pending_(winsz_)
  , max_tasks_(max_tasks)
  , promise_(std::move(pr))
  , trace_(internal::trace_name) {
  // generate the slots
  for (auto&& i : range(detail::n_types)) {
    // FMPI_DBG(nslots_[i]);
    slots_[i] = tlx::RingBuffer<int>{nslots_[i]};
  }

  reset_slots();
}

ScheduleCtx::ScheduleCtx(
    std::array<std::size_t, detail::n_types> nslots, collective_promise pr)
  : ScheduleCtx(
        nslots, std::move(pr), std::numeric_limits<std::size_t>::max()) {
}

inline void ScheduleCtx::reset_slots() {
  std::size_t n = 0;
  for (auto&& i : range(detail::n_types)) {
    slots_[i].clear();
    std::generate_n(
        std::front_inserter(slots_[i]), nslots_[i], [&n]() { return n++; });
  }
}

inline void ScheduleCtx::complete_some() {
  steady_timer timer{trace_.duration(internal::t_complete_some)};

  FixedVector<MPI_Status> statuses(handles_.size());
  std::vector<int>        idxs_completed(handles_.size());
  int                     n = MPI_UNDEFINED;

  auto const ret = MPI_Waitsome(
      static_cast<int>(handles_.size()),
      handles_.data(),
      &n,
      idxs_completed.data(),
      statuses.data());

  FMPI_ASSERT(ret == MPI_SUCCESS);

  idxs_completed.resize((n == MPI_UNDEFINED) ? 0 : n);

  std::array<std::vector<Message>, detail::n_types> msgs;

  for (auto&& idx : idxs_completed) {
    handles_[idx]   = MPI_REQUEST_NULL;
    auto&      task = pending_[idx];
    auto const tid  = rtlx::to_underlying(task.type);
    FMPI_ASSERT(task.valid());
    task.reset();
    slots_[tid].push_front(idx);
    msgs[tid].emplace_back(task.message);
  }

  for (auto&& tid : range(detail::n_types)) {
    if ((not msgs[tid].empty()) and callbacks_[tid]) {
      callbacks_[tid](msgs[tid]);
    }
  }

  n_processed_ += idxs_completed.size();

  if (n_processed_ == max_tasks_) {
    notify_ready();
  }
}

inline void ScheduleCtx::test_all() {
  steady_timer            timer{trace_.duration(internal::t_test_all)};
  FixedVector<MPI_Status> statuses(handles_.size());

  std::vector<int> idxs;
  idxs.reserve(handles_.size());

  auto r = range(handles_.size());

  std::copy_if(
      std::begin(r),
      std::end(r),
      std::back_inserter(idxs),
      [this](auto const& idx) { return handles_[idx] != MPI_REQUEST_NULL; });

  std::array<std::vector<Message>, detail::n_types> msgs;

  int flag = 0;
  for (auto&& idx : idxs) {
    // FMPI_DBG(idx);
    auto const ret = MPI_Test(&handles_[idx], &flag, &statuses[idx]);
    FMPI_ASSERT(ret == MPI_SUCCESS);
    if (flag != 0) {
      handles_[idx]   = MPI_REQUEST_NULL;
      auto&      task = pending_[idx];
      auto const tid  = rtlx::to_underlying(task.type);
      FMPI_ASSERT(task.valid());
      task.reset();
      slots_[tid].push_front(idx);
      msgs[tid].emplace_back(task.message);
      n_processed_++;
    }
  }

  for (auto&& tid : range(detail::n_types)) {
    if ((not msgs[tid].empty()) and callbacks_[tid]) {
      callbacks_[tid](msgs[tid]);
    }
  }

  if (n_processed_ == max_tasks_) {
    notify_ready();
  }
}

inline void ScheduleCtx::notify_ready() {
  FMPI_DBG("notify_ready");
  if (std::exchange(state_, status::ready) == status::pending) {
    promise_.set_value(MPI_SUCCESS);
  }
}

inline void ScheduleCtx::complete_all() {
  steady_timer timer{trace_.duration(internal::t_complete_all)};

  FixedVector<MPI_Status> statuses(handles_.size());

  std::vector<std::size_t> idxs;
  idxs.reserve(handles_.size());

  auto r = range(handles_.size());

  std::copy_if(
      std::begin(r),
      std::end(r),
      std::back_inserter(idxs),
      [this](auto const& idx) { return handles_[idx] != MPI_REQUEST_NULL; });

  if (idxs.empty()) {
    for (auto&& i : range(detail::n_types)) {
      // sanity check
      FMPI_ASSERT(slots_[i].size() == nslots_[i]);
    }
    return;
  }

  auto const ret = MPI_Waitall(
      static_cast<int>(handles_.size()), handles_.data(), statuses.data());
  FMPI_ASSERT(ret == MPI_SUCCESS);

  std::array<std::vector<Message>, detail::n_types> msgs;

  for (auto&& idx : idxs) {
    auto&      task = pending_[idx];
    auto const tid  = rtlx::to_underlying(task.type);
    FMPI_ASSERT(task.valid());

    msgs[tid].emplace_back(task.message);
    task.reset();
  }

  for (auto&& tid : range(detail::n_types)) {
    if ((not msgs[tid].empty()) and callbacks_[tid]) {
      callbacks_[tid](msgs[tid]);
    }
  }

  n_processed_ += idxs.size();

  if (n_processed_ == max_tasks_) {
    notify_ready();
  }

  reset_slots();
}

void ScheduleCtx::register_signal(message_type type, signal&& callable) {
  FMPI_ASSERT(not signals_[rtlx::to_underlying(type)]);
  signals_[rtlx::to_underlying(type)] = std::move(callable);
}

void ScheduleCtx::register_callback(message_type type, callback&& callable) {
  FMPI_ASSERT(not callbacks_[rtlx::to_underlying(type)]);
  callbacks_[rtlx::to_underlying(type)] = std::move(callable);
}

CommDispatcher::CommDispatcher()
  : channel_(internal::channel_capacity) {
  // FMPI_DBG("contructing dispatcher");
  thread_ = std::thread([this]() { worker(); });

  auto const& config      = Pinning::instance();
  auto const  pin_success = pinThreadToCore(thread_, config.dispatcher_core);
  FMPI_ASSERT(pin_success);
}

CommDispatcher::~CommDispatcher() {
  channel_.close();
  thread_.join();
}

ScheduleHandle CommDispatcher::submit(std::unique_ptr<ScheduleCtx> ctx) {
  auto hdl = ScheduleHandle{internal::last_schedule_id++};
  schedules_.assign(hdl, std::move(ctx));
  return hdl;
}

void CommDispatcher::worker() {
  using namespace std::chrono_literals;
  for (;;) {
    // fetch new task, however, wait at most 1us
    CommTask task;
    auto     status = channel_.pop(task, 1ns);

    // FMPI_DBG(sched_getcpu());

    if (status == channel_op_status::closed) {
      progress_all(true);
      break;
    }
    if (status == channel_op_status::success) {
      FMPI_ASSERT(task.valid());

      // retrieve ctx

      auto [it, ok] = schedules_.find(task.id);

      FMPI_ASSERT(ok);

      auto& uptr = it->second;

      if (task.type == message_type::COMMIT ||
          task.type == message_type::BARRIER ||
          task.type == message_type::WAITSOME) {
        // barriers
        if (task.type == message_type::WAITSOME) {
          uptr->complete_some();
        } else {
          uptr->complete_all();
        }

        if (task.type == message_type::COMMIT) {
          FMPI_DBG_STREAM("Commit " << task.id.id());

          auto up = std::move(uptr);
          schedules_.erase(it);
          up->notify_ready();
        }
      } else if (task.type == message_type::COMMIT_ALL) {
        FMPI_DBG_STREAM("Commit all " << task.id.id());
        auto const [first, last] = progress_all(true);

        FixedVector<std::unique_ptr<ScheduleCtx>> uptrs(
            std::distance(first, last));

        std::size_t idx = 0;
        for (auto it = first; it != last; ++it, ++idx) {
          uptrs[idx] = std::move(it->second);
        }

        schedules_.erase(first, last);

        for (auto&& up : uptrs) {
          up->notify_ready();
        }
      } else if (task.type == message_type::ISENDRECV) {
#if 1
        task.type = message_type::IRECV;
        uptr->dispatch_task(task);
        task.type = message_type::ISEND;
        uptr->dispatch_task(task);
        uptr->test_all();
#else

        FMPI_ASSERT(uptr->state_ == ScheduleCtx::status::pending);
        uptr->dispatch_task(task);
#endif
      } else {
        // FMPI_DBG(task.id.id());
        // FMPI_DBG(int(rtlx::to_underlying(task.type)));
        // FMPI_DBG(uptr->nslots_);
        // FMPI_DBG(std::make_pair(uptr->max_tasks_, uptr->n_processed_));
        FMPI_ASSERT(uptr->state_ == ScheduleCtx::status::pending);
        uptr->dispatch_task(task);
        // uptr->test_all();
      }
    }
    if (schedules_.size() > 1) {
      // progress_all();
    }
  }
}

int ScheduleCtx::complete_any(message_type type) {
  // complete one of pending requests and replace it with new slot

  steady_timer timer{trace_.duration(internal::t_complete_any)};

  MPI_Status status{};

  auto const ti = rtlx::to_underlying(type);

  FMPI_ASSERT(ti < detail::n_types);

  auto const first_slot =
      std::accumulate(nslots_.begin(), nslots_.begin() + ti, 0);

  if (nslots_[ti] == 1) {
    FMPI_ASSERT(handles_[first_slot] != MPI_REQUEST_NULL);
    FMPI_ASSERT(pending_[first_slot].valid());
    FMPI_ASSERT(pending_[first_slot].type == type);

    auto const ret = MPI_Wait(&handles_[first_slot], &status);
    FMPI_ASSERT(ret == MPI_SUCCESS);
    handles_[first_slot] = MPI_REQUEST_NULL;
    return first_slot;
  }

  std::vector<MPI_Request> reqs;
  std::vector<int>         idxs;

  auto const last_slot = first_slot + static_cast<int>(nslots_[ti]);
  FMPI_DBG(std::make_pair(first_slot, last_slot));

  for (auto&& idx : range<int>(first_slot, last_slot)) {
    FMPI_ASSERT(pending_[idx].valid());
    FMPI_ASSERT(pending_[idx].type == type);
    FMPI_ASSERT(handles_[idx] != MPI_REQUEST_NULL);
    reqs.emplace_back(handles_[idx]);
    idxs.emplace_back(idx);
  }

  auto const count =
      std::count(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  // (reqs.size() > 0) -> (count == 0)
  FMPI_ASSERT(reqs.empty() || count == 0);

  FMPI_DBG(std::make_pair(reqs.size(), int(ti)));

  int        c = MPI_UNDEFINED;
  auto const ret =
      MPI_Waitany(static_cast<int>(reqs.size()), reqs.data(), &c, &status);

  FMPI_ASSERT(ret == MPI_SUCCESS);
  FMPI_ASSERT(c != MPI_UNDEFINED);

  auto slot      = idxs[c];
  handles_[slot] = MPI_REQUEST_NULL;

  auto& task = pending_[slot];
  FMPI_ASSERT(task.valid());

  if (callbacks_[ti]) {
    callbacks_[ti](std::vector<Message>({task.message}));
  }

  task.reset();

  return slot;
}

int ScheduleCtx::acquire_slot(message_type type) {
  auto const ti = rtlx::to_underlying(type);
  auto&      rb = slots_[ti];

  if (rb.empty()) {
    return complete_any(type);
  }
  // obtain free slot from ring buffer
  auto const slot = rb.back();
  rb.pop_back();
  return slot;
}

void ScheduleCtx::release_slot(message_type type, int slot) {
  auto const ti     = rtlx::to_underlying(type);
  handles_.at(slot) = MPI_REQUEST_NULL;
  slots_[ti].push_front(slot);
}

void ScheduleCtx::dispatch_sendrecv(CommTask task) {
  auto& message = task.message;

  auto const slot_recv = acquire_slot(message_type::IRECV);
  auto const slot_send = acquire_slot(message_type::ISEND);

  auto& signal_send = signals_[rtlx::to_underlying(message_type::ISEND)];
  auto& signal_recv = signals_[rtlx::to_underlying(message_type::IRECV)];

  if (signal_send) {
    signal_send(message);
  }

  if (signal_recv) {
    signal_recv(message);
  }

  auto& rsend = handles_.at(slot_send);
  auto& rrecv = handles_.at(slot_recv);

  FMPI_ASSERT(rsend == MPI_REQUEST_NULL);
  FMPI_ASSERT(rrecv == MPI_REQUEST_NULL);

  auto reqs = std::array<MPI_Request*, 2>{&rsend, &rrecv};
  FMPI_DBG(reqs);

  auto const nslots = nslots_[rtlx::to_underlying(message_type::ISEND)] +
                      nslots_[rtlx::to_underlying(message_type::IRECV)];

  auto ret = handler_.sendrecv(message, reqs, nslots == 2);

  if (rsend == MPI_REQUEST_NULL) {
    release_slot(message_type::ISEND, slot_send);
  } else {
    pending_[slot_send]      = task;
    pending_[slot_send].type = message_type::ISEND;
  }

  if (rrecv == MPI_REQUEST_NULL) {
    release_slot(message_type::IRECV, slot_recv);
  } else {
    pending_[slot_recv]      = task;
    pending_[slot_recv].type = message_type::IRECV;
  }

  FMPI_ASSERT(ret == MPI_SUCCESS);
}

void ScheduleCtx::dispatch_task(CommTask task) {
  if (task.type == message_type::COPY) {
    steady_timer timer{trace_.duration(internal::t_copy)};
    MPI_Request  dummy = MPI_REQUEST_NULL;

    auto ret = handler_(message_type::COPY, task.message, dummy);
    FMPI_ASSERT(ret == MPI_SUCCESS);

    auto const ti = rtlx::to_underlying(message_type::IRECV);
    if (callbacks_[ti]) {
      callbacks_[ti](std::vector<Message>({task.message}));
    }
    return;
  }

  steady_timer timer{trace_.duration(internal::t_dispatch)};

  if (task.type == message_type::ISENDRECV) {
    dispatch_sendrecv(task);
    return;
  }

  auto const slot = acquire_slot(task.type);
  auto const ti   = rtlx::to_underlying(task.type);

  FMPI_ASSERT(handles_.at(slot) == MPI_REQUEST_NULL);

  // Issue new message
  if (signals_[ti]) {
    signals_[ti](task.message);
  }

  auto& req = handles_[slot];
  auto  ret = handler_(task.type, task.message, req);

  FMPI_ASSERT(ret == MPI_SUCCESS);

  if (req == MPI_REQUEST_NULL) {
    release_slot(task.type, slot);
    return;
  }

  std::swap(task, pending_[slot]);

  FMPI_ASSERT(not task.valid());

  // task holds now the previous task...
  // so let's complete callbacks for it

#if 0
  int flag = 0;
  MPI_Test(&handles_[slot], &flag, MPI_STATUS_IGNORE);

  if (flag != 0) {
    handles_[slot] = MPI_REQUEST_NULL;
    if (callbacks_[ti]) {
      callbacks_[ti](std::vector<Message>({pending_[slot].message}));
    }

    rb.push_front(slot);
    pending_[slot].reset();

    n_processed_++;
    if (n_processed_ == max_tasks_) {
      notify_ready();
    }
  }
#endif
}

void CommDispatcher::commit(ScheduleHandle const& hdl) {
  schedule(hdl, message_type::COMMIT);
}

std::
    pair<CommDispatcher::ctx_map::iterator, CommDispatcher::ctx_map::iterator>
    CommDispatcher::progress_all(bool blocking) {
  // all requests
  std::vector<MPI_Request>  reqs;
  std::vector<ScheduleCtx*> sps;
  // map indices in reqs to tuples of (sps, req_idx)
  std::vector<std::pair<std::size_t, std::size_t>> ctx_handles;

  // create a copy of all known schedules
  auto known_schedules = schedules_.known_schedules();
  {
    auto const [first, last] = known_schedules;
    auto const n_schedules   = std::distance(first, last);
    sps.reserve(n_schedules);

    for (auto it = first; it != last; ++it) {
      auto* uptr = it->second.get();
      auto& sp   = sps.emplace_back(uptr);

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
    return known_schedules;
  }

  FMPI_ASSERT(reqs.size() == ctx_handles.size());

  std::vector<int>        idxs_completed(reqs.size());
  FixedVector<MPI_Status> statuses(reqs.size());

  if (blocking) {
    auto const ret = MPI_Waitall(
        static_cast<int>(reqs.size()), reqs.data(), statuses.data());
    FMPI_ASSERT(ret == MPI_SUCCESS);
    std::iota(std::begin(idxs_completed), std::end(idxs_completed), 0);
  } else {
    int n = MPI_UNDEFINED;

    auto const ret = MPI_Testsome(
        static_cast<int>(reqs.size()),
        reqs.data(),
        &n,
        idxs_completed.data(),
        statuses.data());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    idxs_completed.resize((n == MPI_UNDEFINED) ? 0 : n);
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

    FMPI_ASSERT(task.valid());

    auto const ti = rtlx::to_underlying(task.type);

    ctx_tasks[sp_idx][ti].emplace_back(task.message);

    task.reset();

    sp->slots_[ti].push_front(static_cast<int>(req_idx));
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
  return known_schedules;
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

void CommDispatcher::ctx_map::erase(iterator first, iterator last) {
  std::lock_guard<std::mutex> lg{mtx_};
  items_.erase(first, last);
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

std::size_t CommDispatcher::ctx_map::size() const noexcept {
  std::lock_guard<std::mutex> lg{mtx_};
  return items_.size();
}

std::
    pair<CommDispatcher::ctx_map::iterator, CommDispatcher::ctx_map::iterator>
    CommDispatcher::ctx_map::known_schedules() {
  std::lock_guard<std::mutex> lg{mtx_};
  return std::make_pair(std::begin(items_), std::end(items_));
}

}  // namespace fmpi
