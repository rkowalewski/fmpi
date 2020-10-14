#include <cstring>
#include <fmpi/Alltoall.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>

namespace fmpi {

namespace detail {

using namespace std::literals::string_view_literals;
//constexpr auto t_copy = "Tcomm.local_copy"sv;

Alltoall::Alltoall(
    const void*         sendbuf_,
    size_t              sendcount_,
    MPI_Datatype        sendtype_,
    void*               recvbuf_,
    std::size_t         recvcount_,
    MPI_Datatype        recvtype_,
    mpi::Context const& comm_,
    ScheduleOpts const& opts_)
  : sendbuf(sendbuf_)
  , sendcount(sendcount_)
  , sendtype(sendtype_)
  , recvbuf(recvbuf_)
  , recvcount(recvcount_)
  , recvtype(recvtype_)
  , comm(comm_)
  , sendrecvtag_(kTagRing)
  , opts(opts_) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(recvtype == sendtype);
  FMPI_DBG(sendrecvtag_);
  MPI_Type_get_extent(recvtype, &recvlb, &recvextent_);
  MPI_Type_get_extent(sendtype, &sendlb, &sendextent_);
}

inline void* Alltoall::recv_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = recvcount * recvextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(recvbuf, offset);
}

inline const void* Alltoall::send_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = sendcount * sendextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(sendbuf, offset);
}

inline void Alltoall::local_copy() {
  auto const& ctx = comm;
  auto*       dst = recv_offset(ctx.rank());
  auto const* src = send_offset(ctx.rank());

  std::memcpy(dst, src, sendcount * sendextent_);
}

collective_future Alltoall::execute() {
  auto const& schedule = opts.schedule;
  auto const& ctx      = comm;

  FMPI_DBG_STREAM("running algorithm " << opts.name);

  // auto         trace = MultiTrace{std::string_view(opts.name)};
  // steady_timer t_schedule{trace.duration(kScheduleTime)};

  if (ctx.size() < 3) {
    auto const me = ctx.rank();
    auto const other =
        static_cast<mpi::Rank>((ctx.size()) == 1 ? me : 1 - me);

    auto request = std::make_unique<MPI_Request>();

    auto ret = MPI_Irecv(
        recv_offset(other),
        recvcount,
        recvtype,
        other,
        sendrecvtag_,
        ctx.mpiComm(),
        request.get());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    MPI_Send(
        send_offset(other),
        sendcount,
        sendtype,
        other,
        sendrecvtag_,
        ctx.mpiComm());

    local_copy();

    return make_mpi_future(std::move(request));
  }

  // intermediate buffer for two pipelines
  // using thread_alloc = ThreadAllocator<std::byte>;
  // auto buf_alloc     = thread_alloc{};

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(opts.winsz);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

#if 0
  schedule_state->register_signal(
      message_type::IRECV,
      [buf_alloc,
       recvextent,
       recvcount = recvcount,
       recvtype  = recvtype](Message& message) mutable {
        auto const nbytes = recvcount * recvextent;
        auto*      buffer = buf_alloc.allocate(nbytes);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, recvcount, recvtype);
      });
#endif

  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = future.arrival_queue()](std::vector<Message> msgs) mutable {
        std::move(
            std::begin(msgs), std::end(msgs), std::back_inserter(*sptr));
      });

  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  auto const rounds =
      std::max(tlx::div_ceil(schedule.phaseCount(), opts.winsz), 1u);

  for (auto&& r : range(rounds)) {
    auto const last = std::min(schedule.phaseCount(), (r + 1) * opts.winsz);

    for (auto&& rr : range(r * opts.winsz, last)) {
      auto const rpeer = schedule.recvRank(rr);
      auto const speer = schedule.sendRank(rr);

      FMPI_DBG(std::make_tuple(rpeer, speer, sendrecvtag_));

      if (rpeer != ctx.rank()) {
        // auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};
        auto recv = Message{
            recv_offset(rpeer),
            recvcount,
            recvtype,
            rpeer,
            sendrecvtag_,
            ctx.mpiComm()};
        dispatcher.schedule(hdl, message_type::IRECV, recv);
      }

      if (speer != ctx.rank()) {
        auto send = Message{
            send_offset(speer),
            sendcount,
            sendtype,
            speer,
            sendrecvtag_,
            ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::ISEND, send);
      }
    }
    if (opts.type == ScheduleOpts::WindowType::fixed) {
      dispatcher.schedule(hdl, message_type::BARRIER);
    }
  }

  {
    // using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;

    // steady_timer t_copy{trace.duration(detail::t_copy)};
    // we temporarily pause t_schedule and run t_copy.
    // scoped_timer_switch switcher{t_schedule, t_copy};

    local_copy();

    future.arrival_queue()->emplace_back(Message{
        recv_offset(ctx.rank()),
        recvcount,
        recvtype,
        ctx.rank(),
        sendrecvtag_,
        ctx.mpiComm()});
  }

  dispatcher.commit(hdl);
  return future;
}

}  // namespace detail
}  // namespace fmpi
