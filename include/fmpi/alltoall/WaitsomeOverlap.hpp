#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Schedule.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <fmpi/util/Trace.hpp>
#include <numeric>
#include <string_view>
#include <utility>

namespace fmpi {

namespace detail {

using namespace std::literals::string_view_literals;
constexpr auto t_initialize = "Tmain.t_initialize"sv;
constexpr auto t_dispatch   = "Tmain.t_dispatch"sv;
constexpr auto t_receive    = "Tmain.t_receive"sv;
constexpr auto t_idle       = "Tmain.t_idle"sv;
constexpr auto t_compute    = kComputationTime;

template <class T, class Allocator>
class Piece;

template <class T>
using simple_vector =
    tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

struct CollectiveArgs {
  const void*         sendbuf;
  std::size_t         sendcount;
  MPI_Datatype        sendtype;
  void*               recvbuf;
  std::size_t         recvcount;
  MPI_Datatype        recvtype;
  mpi::Context const& comm;

  CollectiveArgs(
      const void*         sendbuf_,
      std::size_t         sendcount_,
      MPI_Datatype        sendtype_,
      void*               recvbuf_,
      std::size_t         recvcount_,
      MPI_Datatype        recvtype_,
      mpi::Context const& comm_);
};

struct ScheduleArgs {
  enum class window_type : uint8_t
  {
    SLIDING,
    FIXED
  };

  std::size_t const winsz{};
  std::string const name;
};

class CollectiveCtx {
 public:
  CollectiveCtx(CollectiveArgs args, ScheduleArgs schedule);

  template <class Schedule>
  collective_future waitsome(Schedule schedule);

  template <class Schedule>
  collective_future waitall(Schedule schedule);

 private:
  CollectiveArgs args_;
  ScheduleArgs   schedule_;
};

template <class T, class Op>
void compute(
    CollectiveArgs const& collective_args,
    collective_future&    future,
    Op&&                  op,
    MultiTrace&           trace);

}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void ring_waitsome_overlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  constexpr auto algorithm_name = std::string_view("WaitsomeOverlap");

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto collective_args = detail::CollectiveArgs{
      &*begin,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      &*out,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      ctx};

  auto schedule_args = detail::ScheduleArgs{
      NReqs,
      std::string{Schedule::NAME} + std::string{algorithm_name} +
          std::to_string(NReqs)

  };

  auto const nr = ctx.size();

  auto trace = MultiTrace{std::string_view(schedule_args.name)};

  FMPI_DBG_STREAM(
      "running algorithm " << schedule_args.name
                           << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  auto coll   = detail::CollectiveCtx{collective_args, schedule_args};
  auto future = coll.waitsome(Schedule{ctx});

  detail::compute<value_type>(
      collective_args, future, std::forward<Op&&>(op), trace);

  {
    steady_timer t_idle{trace.duration(detail::t_idle)};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    // comm_dispatcher.loop_until_done();
    auto ret = future.get();
    FMPI_ASSERT(ret == MPI_SUCCESS);
  }

  // auto const& dispatcher_stats = comm_dispatcher.stats();
  // trace.merge(dispatcher_stats.begin(), dispatcher_stats.end());

  // auto const recv_stats = data_channel->statistics();
  // auto const comm_stats = comm_channel->statistics();

  // trace.duration("Tcomm.enqueue") = comm_stats.enqueue_time;
  // trace.duration("Tcomm.dequeue") = comm_stats.dequeue_time;
  // trace.duration("Tcomp.enqueue") = recv_stats.enqueue_time;
  // trace.duration("Tcomp.dequeue") = recv_stats.dequeue_time;
}

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void ring_waitall_overlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  constexpr auto algorithm_name = std::string_view("WaitsomeOverlap");

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto collective_args = detail::CollectiveArgs{
      &*begin,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      &*out,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      ctx};

  auto schedule_args = detail::ScheduleArgs{
      NReqs,
      std::string{Schedule::NAME} + std::string{algorithm_name} +
          std::to_string(NReqs)

  };

  auto const nr = ctx.size();

  auto trace = MultiTrace{std::string_view(schedule_args.name)};

  FMPI_DBG_STREAM(
      "running algorithm " << schedule_args.name
                           << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  auto coll   = detail::CollectiveCtx{collective_args, schedule_args};
  auto future = coll.waitall(Schedule{ctx});

  detail::compute<value_type>(
      collective_args, future, std::forward<Op&&>(op), trace);

  {
    steady_timer t_idle{trace.duration(detail::t_idle)};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    // comm_dispatcher.loop_until_done();
    auto ret = future.get();
    FMPI_ASSERT(ret == MPI_SUCCESS);
  }

  // auto const& dispatcher_stats = comm_dispatcher.stats();
  // trace.merge(dispatcher_stats.begin(), dispatcher_stats.end());

  // auto const recv_stats = data_channel->statistics();
  // auto const comm_stats = comm_channel->statistics();

  // trace.duration("Tcomm.enqueue") = comm_stats.enqueue_time;
  // trace.duration("Tcomm.dequeue") = comm_stats.dequeue_time;
  // trace.duration("Tcomp.enqueue") = recv_stats.enqueue_time;
  // trace.duration("Tcomp.dequeue") = recv_stats.dequeue_time;
}

#if 0
template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 1>
inline void ring_waitall_overlap(
    InputIt               begin,
    OutputIt              out,
    int                   blocksize,
    ::mpi::Context const& ctx,
    Op&&                  op) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;
  using buffer_t   = ::tlx::
      SimpleVector<value_type, ::tlx::SimpleVectorMode::NoInitNoDestroy>;

  constexpr auto algorithm_name = std::string_view("Waitall");

  auto me = ctx.rank();
  auto nr = ctx.size();

  auto schedule = Schedule{ctx};

  auto const name = std::string{Schedule::NAME} +
                    std::string{algorithm_name} + std::to_string(NReqs);

  auto trace = MultiTrace{std::string_view(name)};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  auto const totalExchanges = nr - 1;
  auto const reqsInFlight   = std::min<std::size_t>(totalExchanges, NReqs);

  auto reqWin = detail::SlidingReqWindow<value_type>{
      reqsInFlight, static_cast<std::size_t>(blocksize)};

  // We can save the local copy and do it with the merge operation itself
  reqWin.pending_pieces().push_back(std::make_pair(
      begin + me * blocksize, begin + me * blocksize + blocksize));

  // auto const phaseCount =
  // static_cast<std::size_t>(schedule.phaseCount(ctx));
  auto const nrounds =
      std::int64_t(tlx::div_ceil(totalExchanges, reqsInFlight));

  std::vector<std::size_t> mergePsum;
  mergePsum.reserve(nrounds + 2);
  mergePsum.push_back(0);

  FMPI_DBG(nrounds);

  std::size_t rphase   = 0;
  std::size_t sphase   = 0;
  auto        mergebuf = out;

  auto const nels = nr * blocksize;
  buffer_t   buffer{nels};

  // auto const nbytes = nels * sizeof(value_type);
  // auto const allFitsInL2Cache = (nbytes < fmpi::CACHELEVEL2_SIZE);
  // constexpr bool allFitsInL2Cache = false;

  std::vector<MPI_Request> reqs(2 * reqsInFlight, MPI_REQUEST_NULL);

  auto moveReqWindow = [schedule, blocksize, &ctx, sbuffer = begin, &reqs](
                           auto initialPhases, auto& reqWin) {
    auto const phaseCount = schedule.phaseCount();
    auto const winsize    = reqWin.winsize();

    std::size_t rphase       = 0;
    std::size_t sphase       = 0;
    std::tie(rphase, sphase) = initialPhases;

    FMPI_DBG(initialPhases);

    for (std::size_t nrreqs = 0; nrreqs < winsize && rphase < phaseCount;
         ++rphase) {
      // t_receive from
      auto recvfrom = schedule.recvRank(rphase);

      if (recvfrom == ctx.rank()) {
        continue;
      }

      FMPI_DBG(recvfrom);

      auto* recvBuf = std::next(reqWin.rbuf(), nrreqs * blocksize);

      FMPI_CHECK_MPI(mpi::irecv(
          recvBuf, blocksize, recvfrom, kTagRing, ctx, &reqs[nrreqs]));

      reqWin.pending_pieces().push_back(
          std::make_pair(recvBuf, std::next(recvBuf, blocksize)));

      ++nrreqs;
    }

    for (std::size_t nsreqs = 0; nsreqs < winsize && sphase < phaseCount;
         ++sphase) {
      auto sendto = schedule.sendRank(sphase);

      if (sendto == ctx.rank()) {
        continue;
      }

      FMPI_DBG(sendto);

      FMPI_CHECK_MPI(mpi::isend(
          std::next(sbuffer, sendto * blocksize),
          blocksize,
          sendto,
          kTagRing,
          ctx,
          &reqs[winsize + nsreqs++]));
    }

    return std::make_pair(rphase, sphase);
  };

  {
    steady_timer t{trace.duration(kCommunicationTime)};

    std::tie(rphase, sphase) =
        moveReqWindow(std::make_pair(rphase, sphase), reqWin);

    FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));

    reqWin.buffer_swap();

    FMPI_ASSERT(reqWin.pending_pieces().empty());
    FMPI_ASSERT(!reqWin.ready_pieces().empty());
  }

  for (auto&& win : range<std::size_t>(nrounds - 1)) {
    std::ignore = win;

    {
      steady_timer t{trace.duration(kCommunicationTime)};
      std::tie(rphase, sphase) =
          moveReqWindow(std::make_pair(rphase, sphase), reqWin);
    }

    {
      steady_timer t{trace.duration(kComputationTime)};
      op(reqWin.ready_pieces(), mergebuf);
      auto const nMerged = reqWin.ready_pieces().size() * blocksize;
      mergebuf += nMerged;
      mergePsum.push_back(mergePsum.back() + nMerged);
      reqWin.ready_pieces().clear();
    }

    {
      steady_timer t{trace.duration(kCommunicationTime)};
      FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));
      reqWin.buffer_swap();
    }
  }

  {
    steady_timer t{trace.duration(kComputationTime)};

    auto mergeSrc = &*out;
    auto target   = buffer.begin();

    // if we have intermediate merge operations then we have already merged
    // chunks in the out buffer and use the dedicated merge buffer as
    // destinated merged buffer. Otherwise, we merge directly into the out
    // buffer.
    FMPI_DBG(mergePsum);

    if (mergePsum.back() > 0) {
      FMPI_DBG(mergePsum);
      std::transform(
          std::begin(mergePsum),
          std::prev(std::end(mergePsum)),
          std::next(std::begin(mergePsum)),
          std::back_inserter(reqWin.ready_pieces()),
          [from = mergeSrc](auto first, auto last) {
            return std::make_pair(
                std::next(from, first), std::next(from, last));
          });
    }

    FMPI_DBG("final merge");
    FMPI_DBG(reqWin.ready_pieces().size());

    op(reqWin.ready_pieces(), target);

    // FMPI_DBG(reqWin.ready_pieces());

    if (target != &*out) {
      std::move(target, target + nels, out);
    }
  }

  trace.value<int64_t>(kCommRounds) = nrounds;
}
#endif

namespace detail {

template <class T, class Allocator>
class Piece {
  using range = gsl::span<T>;
  range      span_{};
  Allocator* alloc_{};

 public:
  using value_type     = T;
  using size_type      = typename range::size_type;
  using iterator       = T*;
  using const_iterator = T const*;

  constexpr Piece() = default;

  constexpr explicit Piece(gsl::span<T> span) noexcept
    : Piece(span, nullptr) {
  }

  constexpr Piece(gsl::span<T> span, Allocator* alloc) noexcept
    : span_(span)
    , alloc_(alloc) {
    FMPI_DBG(span.size());
  }

  ~Piece() {
    if (alloc_) {
      alloc_->deallocate(span_.data(), span_.size());
    }
  }

  Piece(Piece const&) = delete;

  constexpr Piece(Piece&& other) noexcept {
    *this = std::move(other);
  }

  Piece& operator=(Piece const&) = delete;

  constexpr Piece& operator=(Piece&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    using std::swap;
    swap(span_, other.span_);
    swap(alloc_, other.alloc_);

    // reset other to null span to avoid double frees
    other.span_  = gsl::span<T>{};
    other.alloc_ = nullptr;

    return *this;
  }

  constexpr iterator data() noexcept {
    return span_.data();
  }

  [[nodiscard]] constexpr const_iterator data() const noexcept {
    return span_.data();
  }

  //! return number of items in range
  [[nodiscard]] constexpr size_type size() const noexcept {
    return span_.size();
  }

  //! return mutable T* to first element
  constexpr iterator begin() noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
    return begin();
  }

  //! return mutable T* beyond last element
  constexpr iterator end() noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  [[nodiscard]] constexpr const_iterator cend() const noexcept {
    return end();
  }
};

CollectiveCtx::CollectiveCtx(CollectiveArgs args, ScheduleArgs schedule)
  : args_(args)
  , schedule_(std::move(schedule)) {
}

CollectiveArgs::CollectiveArgs(
    const void*         sendbuf_,
    std::size_t         sendcount_,
    MPI_Datatype        sendtype_,
    void*               recvbuf_,
    std::size_t         recvcount_,
    MPI_Datatype        recvtype_,
    mpi::Context const& comm_)
  : sendbuf(sendbuf_)
  , sendcount(sendcount_)
  , sendtype(sendtype_)
  , recvbuf(recvbuf_)
  , recvcount(recvcount_)
  , recvtype(recvtype_)
  , comm(comm_) {
}

#if 0

const void*         sendbuf;
  std::size_t         sendcount;
  MPI_Datatype        sendtype;
  void*               recvbuf;
  std::size_t         recvcount;
  MPI_Datatype        recvtype;
  mpi::Context const& comm;
#endif

template <class Schedule>
collective_future CollectiveCtx::waitsome(Schedule schedule) {
  MPI_Aint recvlb{};
  MPI_Aint recvextent{};
  MPI_Aint sendlb{};
  MPI_Aint sendextent{};
  MPI_Type_get_extent(args_.recvtype, &recvlb, &recvextent);
  MPI_Type_get_extent(args_.sendtype, &sendlb, &sendextent);

  auto const& ctx = args_.comm;

  auto trace = MultiTrace{std::string_view(schedule_.name)};

  FMPI_DBG_STREAM(
      "running algorithm " << schedule_.name
                           << ", sendcount: " << args_.sendcount);

  steady_timer t_init{trace.duration(detail::t_initialize)};

  // exclude the local message to MPI_Self
  auto const n_rounds     = schedule.phaseCount();
  auto const reqsInFlight = std::min(std::size_t(n_rounds), schedule_.winsz);

  // intermediate buffer for two pipelines
  using thread_alloc = ThreadAllocator<std::byte>;

  auto buf_alloc = thread_alloc{};

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(reqsInFlight);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

  schedule_state->register_signal(
      message_type::IRECV,
      [buf_alloc,
       recvextent,
       recvcount = args_.recvcount,
       recvtype  = args_.recvtype](Message& message) mutable {
        auto const nbytes = recvcount * recvextent;
        auto*      buffer = buf_alloc.allocate(nbytes);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, recvcount, recvtype);
      });

  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = future.arrival_queue()](std::vector<Message> msgs) mutable {
        FMPI_ASSERT(sptr);
        std::move(
            std::begin(msgs), std::end(msgs), std::back_inserter(*sptr));
      });

  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  // auto future = collective_future{std::move(schedule_state), queue};

  t_init.finish();

  {
    FMPI_DBG("Sending essages");
    steady_timer t_dispatch{trace.duration(detail::t_dispatch)};

    for (auto&& r : range(schedule.phaseCount())) {
      auto const rpeer = schedule.recvRank(r);
      auto const speer = schedule.sendRank(r);

      if (rpeer != ctx.rank()) {
        auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::IRECV, recv);
      }

      if (speer != ctx.rank()) {
        auto const  offset = speer * args_.sendcount * sendextent;
        auto const* sbuf   = fmpi::detail::add(args_.sendbuf, offset);
        auto        send   = Message{
            const_cast<void*>(sbuf),
            args_.sendcount,
            args_.sendtype,
            speer,
            kTagRing,
            ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::ISEND, send);
      }
    }

    dispatcher.commit(hdl);
  }

  return future;
}

template <class Schedule>
collective_future CollectiveCtx::waitall(Schedule schedule) {
  MPI_Aint recvlb{};
  MPI_Aint recvextent{};
  MPI_Aint sendlb{};
  MPI_Aint sendextent{};
  MPI_Type_get_extent(args_.recvtype, &recvlb, &recvextent);
  MPI_Type_get_extent(args_.sendtype, &sendlb, &sendextent);

  auto const& ctx = args_.comm;

  auto trace = MultiTrace{std::string_view(schedule_.name)};

  FMPI_DBG_STREAM(
      "running algorithm " << schedule_.name
                           << ", sendcount: " << args_.sendcount);

  steady_timer t_init{trace.duration(detail::t_initialize)};

  // exclude the local message to MPI_Self
  auto const n_rounds     = schedule.phaseCount();
  auto const reqsInFlight = std::min(std::size_t(n_rounds), schedule_.winsz);

  // intermediate buffer for two pipelines
  using thread_alloc = ThreadAllocator<std::byte>;

  auto buf_alloc = thread_alloc{};

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(reqsInFlight);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto queue   = future.arrival_queue();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

  schedule_state->register_signal(
      message_type::IRECV,
      [buf_alloc,
       recvextent,
       recvcount = args_.recvcount,
       recvtype  = args_.recvtype](Message& message) mutable {
        auto const nbytes = recvcount * recvextent;
        auto*      buffer = buf_alloc.allocate(nbytes);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, recvcount, recvtype);
      });

  schedule_state->register_callback(
      message_type::IRECV, [sptr = queue](std::vector<Message> msgs) mutable {
        FMPI_ASSERT(sptr);
        std::move(
            std::begin(msgs), std::end(msgs), std::back_inserter(*sptr));
      });

  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  // auto future = collective_future{std::move(schedule_state), queue};

  t_init.finish();

  {
    FMPI_DBG("Sending essages");
    steady_timer t_dispatch{trace.duration(detail::t_dispatch)};

    for (auto&& r : range(schedule.phaseCount())) {
      auto const rpeer = schedule.recvRank(r);
      auto const speer = schedule.sendRank(r);

      if (rpeer != ctx.rank()) {
        auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::IRECV, recv);
      }

      if (speer != ctx.rank()) {
        auto const  offset = speer * args_.sendcount * sendextent;
        auto const* sbuf   = fmpi::detail::add(args_.sendbuf, offset);
        auto        send   = Message{
            const_cast<void*>(sbuf),
            args_.sendcount,
            args_.sendtype,
            speer,
            kTagRing,
            ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::ISEND, send);
      }
    }

    dispatcher.commit(hdl);
  }

  return future;
}

template <class T, class Op>
void compute(
    CollectiveArgs const& collective_args,
    collective_future&    future,
    Op&&                  op,
    MultiTrace&           trace) {
  using value_type = T;
  // intermediate buffer for two pipelines
  using thread_alloc = ThreadAllocator<value_type>;

  using iter_pair = std::pair<value_type*, value_type*>;
  using piece     = detail::Piece<value_type, thread_alloc>;
  using chunk     = std::variant<piece, detail::simple_vector<value_type>>;
  using pieces_t  = std::vector<chunk>;

  auto const& ctx         = collective_args.comm;
  auto const  blocksize   = collective_args.recvcount;
  auto const  nels        = ctx.size() * blocksize;
  auto const  n_exchanges = ctx.size() - 1;
  auto        queue       = future.arrival_queue();
  auto        buf_alloc   = thread_alloc{};

  auto const* begin = static_cast<value_type const*>(collective_args.sendbuf);
  auto*       out   = static_cast<value_type*>(collective_args.recvbuf);

  pieces_t pieces;
  pieces.reserve(ctx.size());

  {
    using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;

    steady_timer t_receive{trace.duration(detail::t_receive)};

    FMPI_DBG("processing message arrivals...");

    // prefix sum over all processed chunks
    auto       d_first = out;
    auto const d_last  = std::next(out, nels);

    auto enough_work =
        [/*&config*/](
            typename pieces_t::const_iterator /*c_first*/,
            typename pieces_t::const_iterator /*c_last*/) -> bool {
      // auto const     npieces    = std::distance(c_first, c_last);
      // constexpr auto min_pieces = 1;

      // auto const nbytes = std::accumulate(
      //    c_first, c_last, std::size_t(0), [](auto acc, auto const& c) {
      //      return acc + std::visit(
      //                       [](auto&& v) -> std::size_t {
      //                         return v.size() * sizeof(value_type);
      //                       },
      //                       c);
      //    });

      // auto const ncpus_rank = std::size_t(config.domain_size);
      return true;

      // return npieces > min_pieces && (nbytes >= (kCacheSizeL2 *
      // ncpus_rank));
    };

    {
      auto local_span = gsl::span{
          std::next(const_cast<value_type*>(begin), ctx.rank() * blocksize),
          std::next(
              const_cast<value_type*>(begin), (ctx.rank() + 1) * blocksize)};

      pieces.emplace_back(piece{local_span});

      // segment task{};
      std::size_t n = n_exchanges;
      // while (data_channel->wait_dequeue(task)) {
      std::vector<Message> msgs;
      msgs.reserve(n);
      while (n) {
        std::size_t m = 0;
        queue->pop_all(std::back_inserter(msgs), m);

        std::transform(
            std::begin(msgs),
            std::end(msgs),
            std::back_inserter(pieces),
            [palloc = &buf_alloc](auto& msg) {
              auto span = gsl::span(
                  static_cast<value_type*>(msg.buffer()), msg.count());

              FMPI_DBG_STREAM(
                  "receiving segment: " << std::make_pair(
                      msg.peer(), span.data()));

              return piece{span, palloc};
            });

        msgs.clear();
        n -= m;

        if (enough_work(pieces.begin(), pieces.end())) {
          steady_timer t_comp{trace.duration(kComputationTime)};
          // we temporarily pause t_receive and run t_comp.
          scoped_timer_switch switcher{t_receive, t_comp};
          // merge all chunks
          std::vector<iter_pair> chunks;
          chunks.reserve(pieces.size());

          std::transform(
              std::begin(pieces),
              std::end(pieces),
              std::back_inserter(chunks),
              [](auto& c) {
                return std::visit(
                    [](auto&& v) -> iter_pair {
                      FMPI_ASSERT(v.begin() <= v.end());
                      return std::make_pair(v.begin(), v.end());
                    },
                    c);
              });

          auto const n_elements = std::accumulate(
              std::begin(chunks),
              std::end(chunks),
              std::size_t(0),
              [](auto acc, auto const& c) {
                return acc + std::distance(c.first, c.second);
              });

          FMPI_DBG(n_elements);

          auto last = d_last;

          using diff_t = typename std::iterator_traits<T*>::difference_type;

          if (diff_t(n_elements) <= std::distance(d_first, d_last)) {
            last = op(chunks, d_first);
            pieces.emplace_back(piece{gsl::span{d_first, last}});
            std::swap(d_first, last);
          } else {
            auto buffer = detail::simple_vector<value_type>{n_elements};
            last        = op(chunks, buffer.begin());

            FMPI_ASSERT(last == buffer.end());

            pieces.emplace_back(std::move(buffer));
          }

          FMPI_DBG_STREAM("clearing " << pieces.size() << "pieces");
          pieces.erase(std::begin(pieces), std::prev(std::end(pieces)));
        }
      }
    }
  }

  {
    steady_timer t_comp{trace.duration(detail::t_compute)};
    FMPI_DBG("final merge");
    FMPI_DBG(pieces.size());

    std::vector<iter_pair> chunks;
    chunks.reserve(pieces.size());

    std::transform(
        std::begin(pieces),
        std::end(pieces),
        std::back_inserter(chunks),
        [](auto& c) {
          return std::visit(
              [](auto&& v) -> iter_pair {
                return std::make_pair(v.begin(), v.end());
              },
              c);
        });

    auto const n_elements = std::accumulate(
        std::begin(chunks),
        std::end(chunks),
        std::size_t(0),
        [](auto acc, auto const& c) {
          return acc + std::distance(c.first, c.second);
        });

    FMPI_DBG(n_elements);
    FMPI_ASSERT(n_elements == nels);

    auto       mergeBuffer = detail::simple_vector<value_type>{nels};
    auto const last        = op(chunks, mergeBuffer.begin());

    FMPI_ASSERT(last == mergeBuffer.end());

    std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  }
}

}  // namespace detail
}  // namespace fmpi
#endif
