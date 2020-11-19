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
  , sendrecvtag(comm.collectiveTag())
  , opts(opts_) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(recvtype == sendtype);
  FMPI_DBG(sendrecvtag);
  MPI_Type_get_extent(recvtype, &recvlb, &recvextent);
  MPI_Type_get_extent(sendtype, &sendlb, &sendextent);
}

inline void* AlltoallCtx::recv_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = recvcount * recvextent;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(recvbuf, offset);
}

inline const void* AlltoallCtx::send_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = sendcount * sendextent;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(sendbuf, offset);
}

inline void AlltoallCtx::local_copy() {
  auto const& ctx = comm;
  auto*       dst = recv_offset(ctx.rank());
  auto const* src = send_offset(ctx.rank());

  std::copy_n(
      static_cast<std::byte const*>(src),
      sendcount * sendextent,
      static_cast<std::byte*>(dst));
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
        sendrecvtag,
        ctx.mpiComm(),
        &future.native_handle());

    FMPI_ASSERT(ret == MPI_SUCCESS);

    MPI_Send(
        send_offset(other),
        static_cast<int>(sendcount),
        sendtype,
        other,
        sendrecvtag,
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

  FMPI_DBG(nslots);

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

#if 1
  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = future.allocate_queue(ctx.size())](
          const std::vector<Message>& msgs) {
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
      sendrecvtag,
      recv_offset(ctx.rank()),
      recvcount,
      recvtype,
      ctx.rank(),
      sendrecvtag,
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
            sendrecvtag,
            recv_offset(rpeer),
            recvcount,
            recvtype,
            rpeer,
            sendrecvtag,
            ctx.mpiComm()};
        type = message_type::ISENDRECV;
      } else if (rpeer != ctx.rank()) {
        // recv
        msg = make_receive(
            recv_offset(rpeer),
            recvcount,
            recvtype,
            rpeer,
            sendrecvtag,
            ctx.mpiComm());
        type = message_type::IRECV;
      } else if (speer != ctx.rank()) {
        // send
        msg = make_send(
            send_offset(speer),
            sendcount,
            sendtype,
            speer,
            sendrecvtag,
            ctx.mpiComm());
        type = message_type::ISEND;
      }

      if (type != message_type::INVALID) {
        dispatcher.schedule(hdl, type, msg);
      }
    }

    if (r < (rounds - 1) and opts.winsz > 1) {
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

    future.arrival_queue()->push(make_receive(
        recv_offset(ctx.rank()),
        recvcount,
        recvtype,
        ctx.rank(),
        sendrecvtag,
        ctx.mpiComm()));
  }
#endif

  dispatcher.commit(hdl);
  return future;
}

class BruckAlgorithm {
  using buffer_t = fmpi::SimpleVector<std::byte>;

  buffer_t          tmpbuf;
  void*             recvbuffer;
  std::size_t const blocksize;
  mpi::Rank const   rank{};
  uint32_t const    size;
  uint32_t const    nrounds;
  mutable uint32_t  round = 0;
  // buffer_t::iterator sbuf;
  // buffer_t::iterator rbuf;

 public:
  BruckAlgorithm(
      void* rbuf, std::size_t blen, mpi::Context const& ctx, uint32_t rounds)
    : tmpbuf(blen * ctx.size())
    , recvbuffer(rbuf)
    , blocksize(blen)
    , rank(ctx.rank())
    , size(ctx.size())
    , nrounds(rounds) {
  }

 public:
  void unpack(Message message) {
    // unpack from recvbuffer to tmpbuf
    // this is actually a memcpy, however, MPI does not provide a way to
    // express memcpy with custom MPI datatypes.
    // Therefore, we send a message to MPI_
    FMPI_CHECK_MPI(MPI_Sendrecv(
        message.recvbuffer(),
        1,
        message.recvtype(),
        rank,
        message.sendtag(),
        tmpbuf.data(),
        1,
        message.recvtype(),
        rank,
        message.recvtag(),
        message.comm(),
        MPI_STATUS_IGNORE));
  }

  bool done() const FMPI_NOEXCEPT {
    round++;
    FMPI_ASSERT(round <= nrounds);
    return round == nrounds;
  }

  void rotate_down() {
    for (auto&& b_src : fmpi::range(size)) {
      auto const my_rank = static_cast<uint32_t>(rank);
      auto const b_dest  = (my_rank - b_src + size) % size;
      auto const offset  = b_src * blocksize;

      auto const first =
          static_cast<std::byte const*>(add(tmpbuf.data(), offset));
      auto const last = first + blocksize;
      auto const d_first =
          static_cast<std::byte*>(add(recvbuffer, b_dest * blocksize));

      std::copy(first, last, d_first);
    }
  }

  buffer_t& buffer() noexcept {
    return tmpbuf;
  }

  buffer_t const& buffer() const noexcept {
    return tmpbuf;
  }
};

collective_future AlltoallCtx::comm_intermediate() {
  constexpr auto r = 2u;
  // w = log_2 n
  /*
   auto const w = static_cast<uint32_t>(std::ceil(fmpi::log(r, comm.size())));
  */
  auto const w = static_cast<uint32_t>(tlx::integer_log2_ceil(comm.size()));

  auto  promise    = collective_promise{};
  auto  future     = promise.get_future();
  auto& dispatcher = static_dispatcher_pool();

  auto const blocksize = sendextent * sendcount;
  auto algo = std::make_shared<BruckAlgorithm>(recvbuf, blocksize, comm, w);
  // algo->sbuf            = algo->tmpbuf.begin() + blocksize * comm.size();
  // algo->rbuf            = algo->sbuf + blocksize * comm.size() / 2;

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(r - 1);
  auto schedule_state =
      std::make_unique<ScheduleCtx>(nslots, std::move(promise));

  schedule_state->register_callback(
      message_type::IRECV, [sptr = algo](const std::vector<Message>& msgs) {
        FMPI_ASSERT(msgs.size() == 1);
        auto const& message = msgs.front();
        FMPI_ASSERT(message.sendtype() == message.recvtype());

        auto const done = sptr->done();

        // note: if we have an associative-decomposable function,
        // unpacking is only needed in communication rounds
        // and the rotate_down can be omitted.
        //
        // otherwise: always unpack, and rotate down in the last round
        sptr->unpack(message);
        if (done) {
          sptr->rotate_down();
        } else {
          // sptr->unpack(message);
        }

        // free message type
        auto my_type = message.recvtype();
        FMPI_CHECK_MPI(MPI_Type_free(&my_type));
      });

  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  {
    // Phase 1: rotate
    // O(p * blocksize)
    auto r_size  = static_cast<mpi::Rank>(comm.size());
    auto first   = static_cast<const std::byte*>(sendbuf);
    auto n_first = static_cast<const std::byte*>(send_offset(comm.rank()));
    auto last    = static_cast<const std::byte*>(send_offset(r_size));

    // rotate leftwards
    std::rotate_copy(first, n_first, last, algo->buffer().data());
  }

  // note: this can be included with associative-decomposable functions
  // local copy
  // std::copy_n(
  //    static_cast<std::byte const*>(send_offset(comm.rank())),
  //    sendcount * sendextent,
  //    static_cast<std::byte*>(recvbuf));

  // auto blocks = fmpi::FixedVector<int>(comm.size());
  auto displs = fmpi::FixedVector<int>(comm.size());
  // auto blens  = fmpi::FixedVector<int>(comm.size());

  MPI_Datatype packed_type{};

  // Phase 2: Communication

  for (int32_t j = 1; j < static_cast<int32_t>(comm.size()); j <<= 1) {
    // for (auto&& d : fmpi::range(1u, r)) {
    // auto const j = static_cast<mpi::Rank>(d * std::pow(r, i));

    // a) pack blocks into a contigous send buffer
    int nblocks = 0;

    for (auto&& idx : range<int32_t>(1, comm.size())) {
      if (idx & j) {
        displs[nblocks] = idx * static_cast<int32_t>(sendcount);
        // blens[nblocks]  = static_cast<int>(sendcount);
        nblocks++;
      }
    }

    FMPI_DBG(nblocks);

    FMPI_CHECK_MPI(MPI_Type_create_indexed_block(
        nblocks,
        static_cast<int>(sendcount),
        displs.data(),
        sendtype,
        &packed_type));

    FMPI_CHECK_MPI(MPI_Type_commit(&packed_type));

    {
      int32_t const r_i      = comm.rank();
      int32_t const s_i      = comm.size();
      int32_t const recvfrom = (r_i - j + s_i) % s_i;
      int32_t const sendto   = (r_i + j) % s_i;

      FMPI_DBG(std::make_pair(recvfrom, sendto));

      auto msg = Message{
          algo->buffer().data(),
          1,
          packed_type,
          static_cast<mpi::Rank>(sendto),
          sendrecvtag,
          recvbuf,
          1,
          packed_type,
          static_cast<mpi::Rank>(recvfrom),
          sendrecvtag,
          comm.mpiComm()};

      dispatcher.schedule(hdl, message_type::ISENDRECV, msg);
    }

    // not needed here because we have winsz == 1
    // dispatcher.schedule(hdl, message_type::BARRIER);
    //}
  }

  dispatcher.commit(hdl);
  return future;
}

}  // namespace detail

collective_future alltoall_tune(
    const void*         sendbuf,
    std::size_t         sendcount,
    MPI_Datatype        sendtype,
    void*               recvbuf,
    std::size_t         recvcount,
    MPI_Datatype        recvtype,
    mpi::Context const& ctx) {
  MPI_Aint extent, lb;
  MPI_Type_get_extent(sendtype, &lb, &extent);

  auto const n = extent * sendcount;
  auto const p = ctx.size();

  constexpr std::size_t kb = 1024;

  std::uint32_t winsz = 0;

  if (p <= 64 and n >= 2 * kb) {
    winsz = 4;
  } else if (p <= 128 and n >= 256) {
    winsz = n > 4 * kb ? 64 : 1;
  } else if (p > 128 and n >= 512) {
    winsz = n > 4 * kb ? 64 : 1;
  }

  if (winsz != 0u) {
    auto           one_factor = OneFactor{ctx};
    constexpr auto win_type   = ScheduleOpts::WindowType::fixed;
    auto           opts       = ScheduleOpts{one_factor, winsz, "", win_type};
    auto           coll       = detail::AlltoallCtx{
        sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        ctx,
        opts};

    return coll.execute();
  }

  auto request = make_mpi_future();

  auto ret = MPI_Ialltoall(
      sendbuf,
      static_cast<int>(sendcount),
      sendtype,
      recvbuf,
      static_cast<int>(recvcount),
      recvtype,
      ctx.mpiComm(),
      &request.native_handle());

  FMPI_ASSERT(ret == MPI_SUCCESS);

  return request;
}
}  // namespace fmpi
