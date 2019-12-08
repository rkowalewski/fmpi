#ifndef ALLTOALL_H
#define ALLTOALL_H

#include <algorithm>
#include <cmath>
#include <fmpi/Constants.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/Memory.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/container/circularfifo.hpp>
#include <fmpi/detail/CommState.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Request.hpp>
#include <future>
#include <memory>
#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>
#include <stack>
#include <string_view>
#include <tlx/math/div_ceil.hpp>
#include <tlx/simple_vector.hpp>
#include <tlx/stack_allocator.hpp>

// Other AllToAll Algorithms

namespace fmpi {

namespace detail {

static constexpr char N_COMM_ROUNDS[] = "Ncomm_rounds";

template <class Schedule, class ReqIdx, class BufAlloc, class CommOp>
inline auto enqueueMpiOps(
    uint32_t   phase,
    mpi::Rank  me,
    uint32_t   reqsInFlight,
    Schedule&& partner,
    ReqIdx&&   reqIdx,
    BufAlloc&& bufAlloc,
    CommOp&&   commOp)
{
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

    auto* buf = bufAlloc(peer, idx);

    RTLX_ASSERT(buf);

    FMPI_CHECK(commOp(buf, peer, idx));

    ++nreqs;
  }

  return phase;
}

template <class InputIt, class OutputIt, class Op>
inline void scatteredPairwise_lt3(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op,
    rtlx::TimeTrace&    trace)
{
  using value_type = typename std::iterator_traits<OutputIt>::value_type;

  using merge_buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto chunks = std::vector<std::pair<InputIt, InputIt>>{};
  chunks.reserve(2);

  auto const me = ctx.rank();

  chunks.emplace_back(
      std::make_pair(begin + me * blocksize, begin + (me + 1) * blocksize));

  if (ctx.size() == 1) {
    trace.tick(MERGE);
    op(chunks, out);
    trace.tock(MERGE);
    return;
  }

  auto other = static_cast<mpi::Rank>(1 - me);

  trace.tick(COMMUNICATION);
  FMPI_CHECK(mpi::sendrecv(
      begin + other * blocksize,
      blocksize,
      other,
      EXCH_TAG_BRUCK,
      out + other * blocksize,
      blocksize,
      other,
      EXCH_TAG_BRUCK,
      ctx));
  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  {
    chunks.emplace_back(std::make_pair(
        out + other * blocksize, out + (other + 1) * blocksize));
    merge_buffer_t buffer{ctx.size() * blocksize};
    op(chunks, buffer.begin());

    std::move(buffer.begin(), buffer.end(), out);
  }

  trace.tock(MERGE);

  trace.put(detail::N_COMM_ROUNDS, 1);

  return;
}

}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void scatteredPairwiseWaitsome(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
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

  auto trace = rtlx::TimeTrace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  trace.tick(COMMUNICATION);

  constexpr auto winsz       = 2 * NReqs;
  constexpr auto mpiReqBytes = winsz * sizeof(MPI_Request);

  using req_buffer_t = SmallVector<MPI_Request, mpiReqBytes>;

  typename req_buffer_t::arena  reqs_arena{};
  typename req_buffer_t::vector reqs(
      winsz, MPI_REQUEST_NULL, typename req_buffer_t::allocator{reqs_arena});

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const totalReqs      = 2 * totalExchanges;
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  using window_buffer =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using lfq_fifo = boost::lockfree::spsc_queue<
      std::pair<
          typename window_buffer::iterator,
          typename window_buffer::iterator>,
      boost::lockfree::capacity<winsz>>;

  auto winbuf = window_buffer{blocksize * winsz};

  auto lfq_chunks = lfq_fifo{};
  // fill freelist
  for (auto&& c : range<std::size_t>(winsz)) {
    while (!lfq_chunks.push(std::make_pair(
        std::next(winbuf.begin(), c * blocksize),
        std::next(winbuf.begin(), (c + 1) * blocksize))))
      ;
  }

  // allocate the communication state which provides the receive buffer
  // enough blocks of size blocksize.
  detail::CommState<value_type, NReqs> commState{};

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs = 0, nrreqs = 0, sphase = 0, rphase = 0;

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&commState, &lfq_chunks](auto /*peer*/, auto reqIdx) {
    typename lfq_fifo::value_type chunk;
    while (!lfq_chunks.pop(chunk))
      ;
    commState.markOccupied(reqIdx, chunk);

    return chunk.first;
  };

  auto receiveOp = [&reqs, blocksize, &ctx](
                       auto* buf, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("receiving from " << peer << " reqIdx " << reqIdx);

    return mpi::irecv(
        buf, blocksize, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  rphase = detail::enqueueMpiOps(
      rphase,
      me,
      reqsInFlight,
      rschedule,
      [](auto nreqs) { return nreqs; },
      rbufAlloc,
      receiveOp);

  nrreqs += reqsInFlight;

  auto sschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.sendRank(ctx, phase);
  };

  auto sbufAlloc = [begin, blocksize](auto peer, auto /*reqIdx*/) {
    return &*std::next(begin, peer * blocksize);
  };

  auto sendOp = [&reqs, blocksize, ctx](auto* buf, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("sending to " << peer << " reqIdx " << reqIdx);
    return mpi::isend(
        buf, blocksize, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  sphase = detail::enqueueMpiOps(
      sphase,
      me,
      reqsInFlight,
      sschedule,
      [reqsInFlight](auto nreqs) { return nreqs + reqsInFlight; },
      sbufAlloc,
      sendOp);

  nsreqs += reqsInFlight;

  auto alreadyDone = reqsInFlight == totalExchanges;

  using iter_pair      = std::pair<InputIt, InputIt>;
  auto chunks_to_merge = std::vector<iter_pair>{};
  chunks_to_merge.reserve(
      alreadyDone ?
                  // we add 1 due to the local portion
          (reqsInFlight + 1)
                  : nr);

  // local portion
  auto f = std::next(begin, me * blocksize);
  auto l = std::next(f, blocksize);
  chunks_to_merge.push_back(std::make_pair(f, l));

  if (alreadyDone) {
    // We are already done
    // Wait for previous round
    FMPI_CHECK(mpi::waitall(&(*reqs.begin()), &(*reqs.end())));

    trace.tock(COMMUNICATION);

    trace.tick(MERGE);

    {
      for (auto&& idx : fmpi::range<int>(0, reqsInFlight)) {
        // mark everything as completed
        chunks_to_merge.emplace_back(commState.markComplete(idx));
      }

      op(chunks_to_merge, out);
    }
    trace.tock(MERGE);
    trace.put(detail::N_COMM_ROUNDS, 1);
    return;
  }

  using index_type = int;
  using indices_buffer_t =
      SmallVector<index_type, winsz * sizeof(index_type)>;

  size_t ncReqs     = 0;
  size_t nrecvTotal = 0;
  auto   outIt      = out;

  std::vector<std::size_t> mergedChunksPsum;
  mergedChunksPsum.reserve(nr);
  mergedChunksPsum.push_back(0);

  typename indices_buffer_t::arena  indices_arena{};
  typename indices_buffer_t::vector indices(
      typename indices_buffer_t::allocator{indices_arena});

  indices.resize(std::distance(reqs.begin(), reqs.end()));

  RTLX_ASSERT(indices.size() <= winsz);

  int n_comm_rounds = 0;

  trace.tock(COMMUNICATION);

#if 0
  using Fifo = CircularFifo<iter_pair, NReqs>;
  static_assert(Fifo::isAlwaysLockFree, "");
  Fifo queue;


  auto consumer = std::async(
      std::launch::async,
      [n = nr,
       blocksize,
       &queue,
       chunks = std::move(chunks_to_merge),
       &op,
       target = out,
       &mergedChunksPsum]() mutable {
        iter_pair chunk;

        std::size_t nmerged = 0;
        while (nmerged < n) {
          while (!queue.pop(chunk)) {
            std::this_thread::yield();
          }

          chunks.emplace_back(chunk);

          if (chunks.size() >= utilization_threshold) {
            op(chunks, target);

            nmerged += chunks.size();
            mergedChunksPsum.push_back(
                mergedChunksPsum.back() + nmerged * blocksize);

            target += nmerged;
          }
        }
      });

#endif

  while (ncReqs < totalReqs) {
    ++n_comm_rounds;
    trace.tick(COMMUNICATION);

    auto* lastIdx =
        mpi::testsome(&(*reqs.begin()), &(*reqs.end()), &(*indices.begin()));

    auto const nCompleted = std::distance(&(*indices.begin()), lastIdx);

    FMPI_DBG(nCompleted);

    ncReqs += nCompleted;

    auto const reqsCompleted =
        std::make_pair(indices.begin(), indices.begin() + nCompleted);

    FMPI_DBG_RANGE(reqsCompleted.first, reqsCompleted.second);

    // we want all receive requests on the left, and send requests on the
    // right
    auto const fstSentIdx = std::partition(
        reqsCompleted.first, reqsCompleted.second, [reqsInFlight](auto req) {
          // left half of reqs array array are receives
          return req < static_cast<int>(reqsInFlight);
        });

    auto const nrecv = std::distance(reqsCompleted.first, fstSentIdx);

    auto const nsent = nCompleted - nrecv;

    FMPI_DBG_STREAM(
        me << " available chunks to merge: "
           << chunks_to_merge.size() + nrecv);

    // mark all receives
    std::for_each(
        reqsCompleted.first,
        fstSentIdx,
        [&commState, &chunks_to_merge](auto reqIdx) {
          auto c = commState.markComplete(reqIdx);
          chunks_to_merge.emplace_back(c);
        });

    nrecvTotal += nrecv;

    auto const nrecvOpen = totalExchanges - nrreqs;

    auto const nslotsRecv = std::min(
        std::min<std::size_t>(nrecv, nrecvOpen), lfq_chunks.read_available());

    FMPI_DBG(nslotsRecv);

    RTLX_ASSERT((reqsCompleted.first + nslotsRecv) <= fstSentIdx);

    rphase = detail::enqueueMpiOps(
        rphase,
        me,
        nslotsRecv,
        rschedule,
        [fstRecvIdx = reqsCompleted.first](auto nreqs) {
          return *std::next(fstRecvIdx, nreqs);
        },
        rbufAlloc,
        receiveOp);

    nrreqs += nslotsRecv;

    auto const nslotsSend =
        std::min<std::size_t>(totalExchanges - nsreqs, nsent);

    sphase = detail::enqueueMpiOps(
        sphase,
        me,
        nslotsSend,
        sschedule,
        [fstSentIdx](auto nreqs) { return *std::next(fstSentIdx, nreqs); },
        sbufAlloc,
        sendOp);

    nsreqs += nslotsSend;

    trace.tock(COMMUNICATION);

    auto const allReceivesDone = nrecvTotal == totalExchanges;

    auto const mergeCount = chunks_to_merge.size();

    // minimum number of chunks to merge: ideally we have a full level2
    // cache
    bool const enough_work = mergeCount >= utilization_threshold;

    FMPI_DBG_STREAM("ready chunks: " << mergeCount);

    if (enough_work || (allReceivesDone && mergeCount)) {
      trace.tick(MERGE);

      FMPI_DBG_STREAM(
          me << " merging " << chunks_to_merge.size() << " chunks");

      // 2) merge all chunks
      op(chunks_to_merge, outIt);

      // 3) increase out iterator
      auto const nmerged = chunks_to_merge.size() * blocksize;

      outIt += nmerged;

      mergedChunksPsum.push_back(mergedChunksPsum.back() + nmerged);

      // 4) reset all completed Chunks
      chunks_to_merge.clear();

      trace.tock(MERGE);
    }
  }

  trace.tick(MERGE);
  trace.put(detail::N_COMM_ROUNDS, n_comm_rounds);

  auto const needsFinalMerge = mergedChunksPsum.size() > 2;
  if (needsFinalMerge) {
    RTLX_ASSERT(chunks_to_merge.empty());

    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto mergeBuffer = merge_buffer_t{std::size_t(nr) * blocksize};
    // generate pairs of chunks to merge
    std::transform(
        std::begin(mergedChunksPsum),
        std::prev(std::end(mergedChunksPsum)),
        std::next(std::begin(mergedChunksPsum)),
        std::back_inserter(chunks_to_merge),
        [rbuf = out](auto first, auto last) {
          return std::make_pair(
              std::next(rbuf, first), std::next(rbuf, last));
        });

    FMPI_DBG(chunks_to_merge.size());
    // merge
    op(chunks_to_merge, mergeBuffer.begin());
    // switch buffer back to output iterator
    std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  }

  trace.tock(MERGE);

  FMPI_DBG(mergedChunksPsum);
}

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 1>
inline void scatteredPairwiseWaitall(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  using value_type = typename std::iterator_traits<OutputIt>::value_type;
  using buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto me = ctx.rank();
  auto nr = ctx.size();

  std::ostringstream os;
  os << Schedule::NAME << "Waitall" << NReqs;
  auto trace = rtlx::TimeTrace{os.str()};

  auto const schedule = Schedule{};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
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
  auto const nrounds = tlx::div_ceil(totalExchanges, reqsInFlight);

  std::vector<std::size_t> mergePsum;
  mergePsum.reserve(nrounds + 2);
  mergePsum.push_back(0);

  FMPI_DBG(nrounds);

  std::size_t rphase = 0, sphase = 0;
  auto        mergebuf = out;

  auto const nels = nr * blocksize;
  buffer_t   buffer{nels};

  // auto const nbytes = nels * sizeof(value_type);
  // auto const allFitsInL2Cache = (nbytes < fmpi::CACHELEVEL2_SIZE);
  // constexpr bool allFitsInL2Cache = false;

  std::vector<MPI_Request> reqs(2 * reqsInFlight, MPI_REQUEST_NULL);

  auto moveReqWindow = [schedule, blocksize, &ctx, sbuffer = begin, &reqs](
                           auto initialPhases, auto& reqWin) {
    auto const phaseCount = schedule.phaseCount(ctx);
    auto const winsize    = reqWin.winsize();

    std::size_t rphase, sphase;
    std::tie(rphase, sphase) = initialPhases;

    for (std::size_t nrreqs = 0; nrreqs < winsize && rphase < phaseCount;
         ++rphase) {
      // receive from
      auto recvfrom = schedule.recvRank(ctx, rphase);

      if (recvfrom == ctx.rank()) continue;

      FMPI_DBG(recvfrom);

      auto* recvBuf = std::next(reqWin.rbuf(), nrreqs * blocksize);

      FMPI_CHECK(mpi::irecv(
          recvBuf, blocksize, recvfrom, EXCH_TAG_RING, ctx, &reqs[nrreqs]));

      reqWin.pending_pieces().push_back(
          std::make_pair(recvBuf, std::next(recvBuf, blocksize)));

      ++nrreqs;
    }

    for (std::size_t nsreqs = 0; nsreqs < winsize && sphase < phaseCount;
         ++sphase) {
      auto sendto = schedule.sendRank(ctx, sphase);

      if (sendto == ctx.rank()) continue;

      FMPI_DBG(sendto);

      FMPI_CHECK(mpi::isend(
          std::next(sbuffer, sendto * blocksize),
          blocksize,
          sendto,
          EXCH_TAG_RING,
          ctx,
          &reqs[winsize + nsreqs++]));
    }

    return std::make_pair(rphase, sphase);
  };

  trace.tick(COMMUNICATION);

  std::tie(rphase, sphase) =
      moveReqWindow(std::make_pair(rphase, sphase), reqWin);

  FMPI_CHECK(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));

  reqWin.buffer_swap();

  RTLX_ASSERT(reqWin.pending_pieces().empty());
  RTLX_ASSERT(!reqWin.ready_pieces().empty());

  trace.tock(COMMUNICATION);

  for (auto&& win : range<std::size_t>(nrounds - 1)) {
    static_cast<void>(win);

    trace.tick(COMMUNICATION);

    std::tie(rphase, sphase) =
        moveReqWindow(std::make_pair(rphase, sphase), reqWin);

    trace.tock(COMMUNICATION);

    if (!reqWin.ready_pieces().empty()) {
      trace.tick(MERGE);
      op(reqWin.ready_pieces(), mergebuf);
      auto const nMerged = reqWin.ready_pieces().size() * blocksize;
      mergebuf += nMerged;
      mergePsum.push_back(mergePsum.back() + nMerged);
      reqWin.ready_pieces().clear();
      trace.tock(MERGE);
    }

    trace.tick(COMMUNICATION);
    FMPI_CHECK(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));
    reqWin.buffer_swap();
    trace.tock(COMMUNICATION);

    FMPI_DBG_RANGE(out, mergebuf);
  }

  trace.tick(MERGE);

  auto mergeSrc = &*out;
  auto target   = buffer.begin();

  // if we have intermediate merge operations then we have already merged
  // chunks in the out buffer and use the dedicated merge buffer as destinated
  // merged buffer.
  // Otherwise, we merge directly into the out buffer.
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

  if (target != &*out) {
    std::move(target, target + nels, out);
  }

  trace.tock(MERGE);

  trace.put(detail::N_COMM_ROUNDS, nrounds);

  FMPI_DBG_RANGE(out, out + nr * blocksize);
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto nr = ctx.size();

  auto trace = rtlx::TimeTrace{"AlltoAll"};

  trace.tick(COMMUNICATION);

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  FMPI_CHECK(mpi::alltoall(
      std::addressof(*begin), blocksize, &rbuf[0], blocksize, ctx));

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = rbuf.get(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, out);

  trace.tock(MERGE);
}
#if 0

template <
    class Schedule,
    bool isBlocking,
    class InputIt,
    class OutputIt,
    class Op>
inline void scatteredPairwise(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto nr = ctx.size();
  auto me = ctx.rank();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  std::ostringstream os;
  os << Schedule::NAME;
  if (isBlocking) {
    os << "Blocking";
  }

  auto trace = rtlx::TimeTrace{os.str()};

  trace.tick(COMMUNICATION);
  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      &rbuf[0] + me * blocksize);

  auto commAlgo = Schedule{};

  std::size_t              nreqs = isBlocking ? 2 : nr * 2;
  std::vector<MPI_Request> reqs(nreqs, MPI_REQUEST_NULL);

  for (int r = 0; r < static_cast<int>(nr); ++r) {
    auto sendto   = commAlgo.sendRank(ctx, r);
    auto recvfrom = commAlgo.recvRank(ctx, r);

    if (sendto == me) {
      sendto = mpi::Rank{};
    }
    if (recvfrom == me) {
      recvfrom = mpi::Rank{};
    }
    if (sendto == MPI_PROC_NULL && recvfrom == MPI_PROC_NULL) {
      continue;
    }

    FMPI_DBG(sendto);
    FMPI_DBG(recvfrom);

    FMPI_CHECK(mpi::irecv(
        std::next(&(rbuf[0]), recvfrom * blocksize),
        blocksize,
        recvfrom,
        EXCH_TAG_RING,
        ctx,
        &reqs[isBlocking ? 0 : r]));

    FMPI_CHECK(mpi::isend(
        std::next(begin, sendto * blocksize),
        blocksize,
        sendto,
        EXCH_TAG_RING,
        ctx,
        &reqs[isBlocking ? 1 : nr + r]));

    if (isBlocking) {
      mpi::waitall(&(*reqs.begin()), &(*reqs.end()));
    }
  }

  if (!isBlocking) {
    mpi::waitall(&(*reqs.begin()), &(*reqs.end()));
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = rbuf.get(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, out);

  trace.tock(MERGE);
}
#endif
}  // namespace fmpi
#endif
