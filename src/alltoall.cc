#include <cstring>
#include <fmpi/Alltoall.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/util/Math.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>

namespace fmpi {
namespace detail {

using namespace std::literals::string_view_literals;
// constexpr auto t_copy = "Tcomm.local_copy"sv;

AlltoallCtx::AlltoallCtx(
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

inline void* AlltoallCtx::recv_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = recvcount * recvextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(recvbuf, offset);
}

inline const void* AlltoallCtx::send_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = sendcount * sendextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(sendbuf, offset);
}

inline void AlltoallCtx::local_copy() {
  auto const& ctx = comm;
  auto*       dst = recv_offset(ctx.rank());
  auto const* src = send_offset(ctx.rank());

  std::memcpy(dst, src, sendcount * sendextent_);
}

collective_future AlltoallCtx::execute() {
  auto const& schedule = opts.schedule;
  auto const& ctx      = comm;

  FMPI_DBG_STREAM("running algorithm " << opts.name);

  // auto         trace = MultiTrace{std::string_view(opts.name)};
  // steady_timer t_schedule{trace.duration(kScheduleTime)};

  if (ctx.size() < 3) {
    auto const me = ctx.rank();
    auto const other =
        static_cast<mpi::Rank>((ctx.size()) == 1 ? me : 1 - me);

    auto future = make_mpi_future();

    auto ret = MPI_Irecv(
        recv_offset(other),
        static_cast<int>(recvcount),
        recvtype,
        other,
        sendrecvtag_,
        ctx.mpiComm(),
        &future.native_handle());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    MPI_Send(
        send_offset(other),
        static_cast<int>(sendcount),
        sendtype,
        other,
        sendrecvtag_,
        ctx.mpiComm());

    local_copy();

    return future;
  }

  if (schedule.is_intermediate()) {
    return comm_intermediate();
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

#if 0
  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = future.allocate_queue(ctx.size())](
          const std::vector<Message>& msgs) mutable {
        for (auto&& msg : msgs) {
          sptr->push(msg);
        }
      });
#endif

  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  auto const rounds = std::max(schedule.phaseCount() / opts.winsz, 1u);

  FMPI_DBG(rounds);

#if 0
  auto msg = Message{
      send_offset(ctx.rank()),
      sendcount,
      sendtype,
      ctx.rank(),
      sendrecvtag_,
      recv_offset(ctx.rank()),
      recvcount,
      recvtype,
      ctx.rank(),
      sendrecvtag_,
      ctx.mpiComm()};

  dispatcher.schedule(hdl, message_type::COPY, msg);
#endif

  Message msg{};

  for (auto&& r : range(rounds)) {
    auto const last = std::min(schedule.phaseCount(), (r + 1) * opts.winsz);

    for (auto&& rr : range(r * opts.winsz, last)) {
      auto const rpeer = schedule.recvRank(rr);
      auto const speer = schedule.sendRank(rr);

      auto type = message_type::INVALID;

      if (rpeer != ctx.rank() and speer != ctx.rank()) {
        // sendrecv
        msg = Message{
            send_offset(speer),
            sendcount,
            sendtype,
            speer,
            sendrecvtag_,
            recv_offset(rpeer),
            recvcount,
            recvtype,
            rpeer,
            sendrecvtag_,
            ctx.mpiComm()};
        type = message_type::ISENDRECV;
      } else if (rpeer != ctx.rank()) {
        // recv
        msg = make_receive(
            recv_offset(rpeer),
            recvcount,
            recvtype,
            rpeer,
            sendrecvtag_,
            ctx.mpiComm());
        type = message_type::IRECV;
      } else if (speer != ctx.rank()) {
        // send
        msg = make_send(
            send_offset(speer),
            sendcount,
            sendtype,
            speer,
            sendrecvtag_,
            ctx.mpiComm());
        type = message_type::ISEND;
      }

      if (type != message_type::INVALID) {
        dispatcher.schedule(hdl, type, msg);
      }
    }

    if (r < (rounds - 1)) {
      FMPI_DBG("scheduling barrier");
      // if this is not the last round
      if (opts.type == ScheduleOpts::WindowType::fixed) {
        dispatcher.schedule(hdl, message_type::BARRIER);
      } else {
        dispatcher.schedule(hdl, message_type::WAITSOME);
      }
    }
  }

#if 1
  // copy
  {
    // using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;

    // steady_timer t_copy{trace.duration(detail::t_copy)};
    // we temporarily pause t_schedule and run t_copy.
    // scoped_timer_switch switcher{t_schedule, t_copy};

    local_copy();

    // future.arrival_queue()->push(make_receive(
    //    recv_offset(ctx.rank()),
    //    recvcount,
    //    recvtype,
    //    ctx.rank(),
    //    sendrecvtag_,
    //    ctx.mpiComm()));
  }
#endif

  dispatcher.commit(hdl);
  return future;
}

struct BruckAlgorithm {
 private:
  using buffer_t = fmpi::FixedVector<std::byte>;

 public:
  BruckAlgorithm(std::size_t bytes)
    : tmpbuf(bytes) {
  }
  int                round = 0;
  buffer_t           tmpbuf;
  buffer_t::iterator sbuf;
  buffer_t::iterator rbuf;
};

collective_future AlltoallCtx::comm_intermediate() {
  constexpr auto r = 2u;
  // w = log_2 n
  auto const w = static_cast<uint32_t>(std::ceil(fmpi::log(r, comm.size())));

  auto  promise    = collective_promise{};
  auto  future     = promise.get_future();
  auto& dispatcher = static_dispatcher_pool();

  auto const blocksize  = sendextent_ * sendcount;
  auto const buffersize = blocksize * comm.size() * 2;
  auto       algo       = std::make_shared<BruckAlgorithm>(buffersize);
  auto       blocks     = fmpi::FixedVector<std::size_t>(comm.size() / 2);
  algo->sbuf            = algo->tmpbuf.begin() + blocksize * comm.size();
  algo->rbuf            = algo->sbuf + blocksize * comm.size() / 2;

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(r - 1);
  auto schedule_state =
      std::make_unique<ScheduleCtx>(nslots, std::move(promise));

  // This is somehow an ugly hack
  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = algo](const std::vector<Message>& /*msgs*/) mutable {
        sptr->round++;
      });

  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  // rotate leftwards
  std::memcpy(
      algo->tmpbuf.data(),
      send_offset(comm.rank()),
      (comm.size() - comm.rank()) * blocksize);

  if (comm.rank() != 0) {
    std::memcpy(
        algo->tmpbuf.data() + (comm.size() - comm.rank()) * blocksize,
        sendbuf,
        comm.rank() * blocksize);
  }

  using fmpi::mod;

  for (auto&& i : fmpi::range(w)) {
    for (auto&& d : fmpi::range(1u, r)) {
      std::ignore = d;
      // auto const j = static_cast<mpi::Rank>(d * std::pow(r, i));
      auto j = static_cast<mpi::Rank>(1 << i);

      // We exchange all blocks where the j-th bit is set
      auto rng = range(1u, comm.size());

      auto const b_last = std::copy_if(
          std::begin(rng), std::end(rng), std::begin(blocks), [j](auto idx) {
            return idx & j;
          });

      // a) pack blocks into a contigous send buffer

      auto const nblocks = std::distance(std::begin(blocks), b_last);

      for (auto&& b : range(nblocks)) {
        auto const block = blocks[b];
        auto       copy  = Message{
            add(algo->tmpbuf.data(), block * blocksize),
            blocksize,
            MPI_BYTE,
            comm.rank(),
            sendrecvtag_,
            add(&*algo->sbuf, b * blocksize),
            blocksize,
            MPI_BYTE,
            comm.rank(),
            sendrecvtag_,
            comm.mpiComm()};

        dispatcher.schedule(hdl, message_type::COPY, copy);
      }

      {
        auto const peers = std::make_pair(
            mod(comm.rank() - j, static_cast<mpi::Rank>(comm.size())),
            mod(comm.rank() + j, static_cast<mpi::Rank>(comm.size())));
        auto const [recvfrom, sendto] = peers;

        FMPI_DBG(peers);

        auto msg = Message{
            &*algo->sbuf,
            nblocks * blocksize,
            MPI_BYTE,
            sendto,
            sendrecvtag_,
            &*algo->rbuf,
            nblocks * blocksize,
            MPI_BYTE,
            recvfrom,
            sendrecvtag_,
            comm.mpiComm()};
        dispatcher.schedule(hdl, message_type::ISENDRECV, msg);
      }

      dispatcher.schedule(hdl, message_type::BARRIER);

      for (auto&& b : range(nblocks)) {
        auto const block = blocks[b];

        auto copy = Message{
            add(&*algo->rbuf, b * blocksize),
            blocksize,
            MPI_BYTE,
            comm.rank(),
            sendrecvtag_,
            add(&*algo->tmpbuf.data(), block * blocksize),
            blocksize,
            MPI_BYTE,
            comm.rank(),
            sendrecvtag_,
            comm.mpiComm()};

        dispatcher.schedule(hdl, message_type::COPY, copy);
      }
    }
  }

  for (auto&& b_src : fmpi::range(static_cast<mpi::Rank>(comm.size()))) {
    auto const b_dest =
        mod(comm.rank() - b_src, static_cast<mpi::Rank>(comm.size()));

    auto const offset = static_cast<std::size_t>(b_src) * blocksize;

    auto copy = make_copy(
        recv_offset(b_dest),
        add(algo->tmpbuf.data(), offset),
        blocksize,
        MPI_BYTE);

    dispatcher.schedule(hdl, message_type::COPY, copy);
  }

  dispatcher.commit(hdl);
  return future;
}

}  // namespace detail
}  // namespace fmpi
