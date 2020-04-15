#ifndef FMPI_ALLTOALL_WAITSOME_HPP
#define FMPI_ALLTOALL_WAITSOME_HPP

#include <cstdint>
#include <numeric>

#include <fmpi/Dispatcher.hpp>

#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/container/StackContainer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/detail/Async.hpp>

#include <rtlx/Trace.hpp>

#include <tlx/container/ring_buffer.hpp>

namespace fmpi {

namespace detail {

template <class Schedule, class ReqIdx, class BufAlloc, class CommOp>
inline auto enqueueMpiOps(
    uint32_t   phase,
    mpi::Rank  me,
    uint32_t   reqsInFlight,
    Schedule&& partner,
    ReqIdx&&   reqIdx,
    BufAlloc&& bufAlloc,
    CommOp&&   commOp) {
  uint32_t nreqs;

  for (nreqs = 0; nreqs < reqsInFlight; ++phase) {
    auto peer = partner(phase);

    if (peer == me) {
      FMPI_DBG_STREAM("skipping local phase " << phase);
      continue;
    }

    auto idx = reqIdx(nreqs);

    FMPI_DBG_STREAM(
        "exchanging data with " << peer << " phase " << phase << " reqIdx "
                                << idx);

    auto buf = bufAlloc(peer, idx);

    FMPI_CHECK_MPI(commOp(buf, peer, idx));

    ++nreqs;
  }

  return phase;
}

template <class InputIteratorIterator, class OutputIterator, class Op>
std::pair<InputIteratorIterator, OutputIterator> apply_compute(
    InputIteratorIterator first,
    InputIteratorIterator last,
    OutputIterator        d_first,
    Op&&                  op) {
  auto const n_merges = std::distance(first, last);

  if (n_merges == 0) {
    return std::make_pair(first, d_first);
  }

  std::vector<
      typename std::iterator_traits<InputIteratorIterator>::value_type>
      chunks;

  chunks.reserve(n_merges);
  std::copy(first, last, std::back_inserter(chunks));

  // 2) merge all chunks
  auto d_last = op(chunks, d_first);

  return std::make_pair(last, d_last);
}
}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void ring_waitsome(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  // Tuning Parameters:

  // NReqs: Maximum Number of pending receives

  // Defines the size of a task. Here we say that at least 50% of pending
  // receives need to be ready before we combine them in a local computation.
  // This of course depends on the size of a pending receive.
  //
  // An alternative may be: Given that messages are relatively small we can
  // wait until either the L2 or L3 cache is full to maximize the benefit for
  // local computation
  constexpr auto utilization_threshold = (NReqs / 2);
  static_assert(
      utilization_threshold > 1, "at least two concurrent receives required");

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::ostringstream os;
  os << Schedule::NAME << "Waitsome" << NReqs;

  auto trace = rtlx::Trace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  rtlx::TimeTrace t_prepare{trace, COMMUNICATION};

  constexpr auto winreqs = 2 * NReqs;

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  using window_buffer =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto winbuf = window_buffer{blocksize * 2 * reqsInFlight};
  using bufit = typename window_buffer::iterator;
  using piece = std::pair<bufit, bufit>;

  using freelist_t = StackContainer<
      tlx::RingBuffer<piece, tlx::StackAllocator<piece, kContainerStackSize>>,
      kContainerStackSize>;

  freelist_t freelist{};
  freelist->allocate(winreqs);
  FMPI_ASSERT(freelist->empty());

  {
    auto r = range<std::size_t>(winreqs);

    std::transform(
        std::begin(r),
        std::end(r),
        std::front_inserter(freelist.container()),
        [begin = std::begin(winbuf), blocksize](auto idx) {
          return std::make_pair(
              std::next(begin, idx * blocksize),
              std::next(begin, (idx + 1) * blocksize));
        });
  }

  StackVector<MPI_Request, winreqs> reqs{};
  reqs->resize(winreqs, MPI_REQUEST_NULL);
  FMPI_ASSERT(reqs->capacity() == winreqs);

  // StackVector<piece, NReqs> arrived_chunks;
  // arrived_chunks->resize(NReqs);
  std::vector<piece> arrived_chunks;
  arrived_chunks.reserve(NReqs + 1);

  arrived_chunks.push_back(std::make_pair(
      std::next(begin, me * blocksize),
      std::next(begin, (me + 1) * blocksize)));

  std::vector<piece> pieces_done;
  pieces_done.reserve(nr);

  StackVector<int, winreqs> indices{};
  indices->resize(winreqs);
  FMPI_ASSERT(indices->capacity() == winreqs);
  // initially we can use the full array of request indices for send and
  // receives
  std::iota(
      indices->begin(), std::next(indices->begin(), reqsInFlight * 2), 0);

  StackVector<piece, NReqs> occupied{};
  occupied->resize(NReqs);

  FMPI_ASSERT(2 * reqsInFlight <= reqs->size());

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&occupied, &freelist](auto /*peer*/, auto reqIdx) {
    auto c = freelist->back();
    freelist->pop_back();
    return (occupied[reqIdx] = c);
  };

  auto receiveOp = [&reqs, &ctx](auto chunk, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("receiving from " << peer << " reqIdx " << reqIdx);

    auto const nels = std::distance(chunk.first, chunk.second);
    return mpi::irecv(
        &*chunk.first, nels, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  auto sschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.sendRank(ctx, phase);
  };

  auto sbufAlloc = [begin, blocksize](auto peer, auto /*reqIdx*/) {
    auto first = std::next(begin, peer * blocksize);
    return std::make_pair(first, std::next(first, blocksize));
  };

  auto sendOp = [&reqs, &ctx](auto chunk, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("sending to " << peer << " reqIdx " << reqIdx);

    auto const nels = std::distance(chunk.first, chunk.second);
    return mpi::isend(
        &*chunk.first, nels, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  std::size_t const total_reqs = 2 * totalExchanges;
  std::size_t       nc_reqs    = 0;
  std::size_t       nsreqs     = 0;
  std::size_t       nrreqs     = 0;
  std::size_t       sphase     = 0;
  std::size_t       rphase     = 0;
  std::size_t       nSlotsRecv = reqsInFlight;
  std::size_t       nSlotsSend = reqsInFlight;
  auto              d_first    = out;

  // Yes, this is an int due to API requirements in our trace module
  int n_comm_rounds = 0;

  // Indices smaller than the pivot are receive requests, then we have send
  // requests
  auto sreqs_pivot = std::next(indices->begin(), reqsInFlight);

  t_prepare.finish();

  do {
    rtlx::TimeTrace t_comm{trace, COMMUNICATION};
    ++n_comm_rounds;

    FMPI_DBG(n_comm_rounds);

    FMPI_DBG("receiving...");
    FMPI_DBG(nSlotsRecv);

    rphase = detail::enqueueMpiOps(
        rphase,
        me,
        nSlotsRecv,
        rschedule,
        [it = indices->begin()](auto nreqs) { return *std::next(it, nreqs); },
        rbufAlloc,
        receiveOp);

    nrreqs += nSlotsRecv;

    FMPI_DBG("sending...");
    FMPI_DBG(nSlotsSend);

    sphase = detail::enqueueMpiOps(
        sphase,
        me,
        nSlotsSend,
        sschedule,
        [it = sreqs_pivot](auto nreqs) { return *std::next(it, nreqs); },
        sbufAlloc,
        sendOp);

    nsreqs += nSlotsSend;

    FMPI_DBG(nc_reqs);

    auto const null_reqs = static_cast<std::size_t>(
        std::count(reqs->begin(), reqs->end(), MPI_REQUEST_NULL));

    auto const n_active_reqs = reqs->size() - null_reqs;
    FMPI_DBG(n_active_reqs);

    // (nc_reqs < total_reqs) $\implies$ (there is at least one non-null)
    FMPI_ASSERT((!(nc_reqs < total_reqs)) || null_reqs < reqs->size());

    int* lastIdx{};
    FMPI_CHECK_MPI(mpi::testsome(
        &(*reqs->begin()),
        &(*reqs->end()),
        &(*indices->begin()),
        MPI_STATUSES_IGNORE,
        lastIdx));

    FMPI_ASSERT(lastIdx >= &*indices->begin());

    auto const nCompleted = std::distance(&(*indices->begin()), lastIdx);

    FMPI_DBG(nCompleted);

    nc_reqs += nCompleted;

    auto const reqsCompleted =
        std::make_pair(indices->begin(), indices->begin() + nCompleted);

    for (auto it = reqsCompleted.first; it < reqsCompleted.second; ++it) {
      reqs[*it] = MPI_REQUEST_NULL;
    }

    FMPI_DBG_RANGE(reqsCompleted.first, reqsCompleted.second);

    // we want all receive requests on the left, and send requests on the
    // right
    sreqs_pivot = std::partition(
        reqsCompleted.first, reqsCompleted.second, [reqsInFlight](auto req) {
          // left half of reqs array array are receives
          return req < static_cast<int>(reqsInFlight);
        });

    auto const nrecv = static_cast<std::size_t>(
        std::distance(reqsCompleted.first, sreqs_pivot));

    FMPI_DBG(nrecv);

    auto const nsent = nCompleted - nrecv;

    FMPI_DBG(nsent);

    nSlotsRecv = std::min<std::size_t>(totalExchanges - nrreqs, nrecv);
    nSlotsSend = std::min<std::size_t>(totalExchanges - nsreqs, nsent);

    FMPI_ASSERT((reqsCompleted.first + nSlotsRecv) <= sreqs_pivot);

    // ensure that we never try to push more than we have available
    // capacity
    FMPI_ASSERT(nrecv <= arrived_chunks.capacity());

    std::transform(
        reqsCompleted.first,
        sreqs_pivot,
        std::back_inserter(arrived_chunks),
        [&occupied](auto reqIdx) { return occupied[reqIdx]; });

    FMPI_DBG(arrived_chunks.size());

    t_comm.finish();

    {
      auto const n =
          std::distance(std::begin(arrived_chunks), std::end(arrived_chunks));

      using diff_type = decltype(n);

      if (n >= static_cast<diff_type>(utilization_threshold)) {
        // trace communication
        rtlx::TimeTrace t_comp{trace, COMPUTATION};

        auto [last_piece, d_last] = detail::apply_compute(
            arrived_chunks.begin(),
            arrived_chunks.end(),
            d_first,
            std::forward<Op>(op));

        // release chunks
        std::move(
            // skip local piece which does not need to be deallocated
            std::next(std::begin(arrived_chunks), (d_first == out)),
            last_piece,
            std::front_inserter(freelist.container()));

        pieces_done.emplace_back(d_first, d_last);
        std::swap(d_first, d_last);

        arrived_chunks.erase(std::begin(arrived_chunks), last_piece);
      }
    }
  } while (nc_reqs < total_reqs);

  trace.put(N_COMM_ROUNDS, n_comm_rounds);

  {
    rtlx::TimeTrace t_comp{trace, COMPUTATION};
    auto const      nels = static_cast<std::size_t>(nr) * blocksize;

    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto mergeBuffer = merge_buffer_t{nels};

    std::copy(
        std::begin(pieces_done),
        std::end(pieces_done),
        std::back_inserter(arrived_chunks));

    FMPI_DBG(arrived_chunks.size());

    auto [last_piece, d_last] = detail::apply_compute(
        std::begin(arrived_chunks),
        std::end(arrived_chunks),
        mergeBuffer.begin(),
        std::forward<Op>(op));

    FMPI_DBG(std::make_pair(d_last, mergeBuffer.end()));
    FMPI_ASSERT(d_last == mergeBuffer.end());

    std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  }
}

}  // namespace fmpi
#endif
