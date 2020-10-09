#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Debug.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
//#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>
#include <numeric>
#include <string_view>
#include <tlx/math/div_ceil.hpp>
#include <utility>

namespace fmpi {

namespace detail {

using namespace std::literals::string_view_literals;
constexpr auto t_initialize = "Tmain.t_initialize"sv;
constexpr auto t_dispatch   = "Tmain.t_dispatch"sv;
constexpr auto t_receive    = "Tmain.t_receive"sv;
constexpr auto t_idle       = "Tmain.t_idle"sv;
constexpr auto t_compute    = kComputationTime;

class Alltoall {
 public:
  Alltoall(
      const void*         sendbuf_,
      std::size_t         sendcount_,
      MPI_Datatype        sendtype_,
      void*               recvbuf_,
      std::size_t         recvcount_,
      MPI_Datatype        recvtype_,
      mpi::Context const& comm_,
      ScheduleOpts const& opts_);

  collective_future execute();

 private:
  [[nodiscard]] const void* send_offset(mpi::Rank r) const;
  [[nodiscard]] void*       recv_offset(mpi::Rank r) const;
  void                      local_copy();

  const void* const   sendbuf;
  std::size_t const   sendcount;
  MPI_Datatype const  sendtype;
  void* const         recvbuf;
  std::size_t const   recvcount;
  MPI_Datatype const  recvtype;
  mpi::Context const& comm;
  MPI_Aint            recvextent_{};
  MPI_Aint            sendextent_{};
  ScheduleOpts const& opts;
};

}  // namespace detail

inline collective_future alltoall(
    const void*         sendbuf,
    std::size_t         sendcount,
    MPI_Datatype        sendtype,
    void*               recvbuf,
    std::size_t         recvcount,
    MPI_Datatype        recvtype,
    mpi::Context const& ctx,
    ScheduleOpts const& schedule_args) {
  auto coll = detail::Alltoall{
      sendbuf,
      sendcount,
      sendtype,
      recvbuf,
      recvcount,
      recvtype,
      ctx,
      schedule_args};
  return coll.execute();
}

namespace detail {

inline Alltoall::Alltoall(
    const void*         sendbuf_,
    std::size_t         sendcount_,
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
  , opts(opts_) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(recvtype == sendtype);
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

inline collective_future Alltoall::execute() {
  auto const& schedule = opts.scheduler;
  auto const& ctx      = comm;

  FMPI_DBG_STREAM("running algorithm " << opts.name);

  if (ctx.size() < 3) {
    auto const me = ctx.rank();
    auto const other =
        static_cast<mpi::Rank>((ctx.size()) == 1 ? me : 1 - me);
    auto ret = MPI_Sendrecv(
        send_offset(other),
        sendcount,
        sendtype,
        other,
        kTagRing,
        recv_offset(other),
        recvcount,
        recvtype,
        other,
        kTagRing,
        ctx.mpiComm(),
        MPI_STATUS_IGNORE);
    local_copy();
    return make_ready_future(ret);
  }

  auto trace = MultiTrace{std::string_view(opts.name)};

  steady_timer t_init{trace.duration(detail::t_initialize)};

  // intermediate buffer for two pipelines
  // using thread_alloc = ThreadAllocator<std::byte>;
  // auto buf_alloc     = thread_alloc{};

  auto const reqsInFlight =
      std::min(std::size_t(schedule.phaseCount()), opts.winsz);

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(reqsInFlight);

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

  t_init.finish();

  {
    auto const winsz = static_cast<uint32_t>(opts.winsz);
    FMPI_DBG("Sending messages");
    steady_timer t_dispatch{trace.duration(detail::t_dispatch)};

    auto const rounds =
        std::max(tlx::div_ceil(schedule.phaseCount(), winsz), 1u);

    for (auto&& r : range(rounds)) {
      auto const last = std::min(schedule.phaseCount(), (r + 1) * winsz);

      for (auto&& rr : range(r * winsz, last)) {
        auto const rpeer = schedule.recvRank(rr);
        auto const speer = schedule.sendRank(rr);

        FMPI_DBG(std::make_pair(rpeer, speer));

        if (rpeer != ctx.rank()) {
          // auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};
          auto recv = Message{
              recv_offset(rpeer),
              recvcount,
              recvtype,
              rpeer,
              kTagRing,
              ctx.mpiComm()};
          dispatcher.schedule(hdl, message_type::IRECV, recv);
        }

        if (speer != ctx.rank()) {
          auto send = Message{
              send_offset(speer),
              sendcount,
              sendtype,
              speer,
              kTagRing,
              ctx.mpiComm()};

          dispatcher.schedule(hdl, message_type::ISEND, send);
        }
      }
      if (opts.type == ScheduleOpts::WindowType::fixed) {
        dispatcher.schedule(hdl, message_type::BARRIER);
      }
    }

    local_copy();

    future.arrival_queue()->emplace_back(Message{
        recv_offset(ctx.rank()),
        recvcount,
        recvtype,
        ctx.rank(),
        kTagRing,
        ctx.mpiComm()});
    dispatcher.commit(hdl);
  }

  return future;
}

}  // namespace detail
}  // namespace fmpi
#endif
