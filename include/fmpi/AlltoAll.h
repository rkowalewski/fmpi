#ifndef ALLTOALL_H
#define ALLTOALL_H

#include <fmpi/Constants.h>
#include <fmpi/Debug.h>
#include <fmpi/Memory.h>
#include <fmpi/NumericRange.h>
#include <fmpi/Schedule.h>
#include <fmpi/detail/CommState.h>
#include <fmpi/mpi/Algorithm.h>
#include <fmpi/mpi/Request.h>
#include <rtlx/Assert.h>
#include <rtlx/Trace.h>

#include <algorithm>
#include <cmath>
#include <memory>
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

  FMPI_DBG_STREAM("running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  trace.tick(COMMUNICATION);

  // std::array<MPI_Request, 2 * NReqs> reqs{};
  // reqs.fill(MPI_REQUEST_NULL);

  constexpr auto nPendingReqs = 2 * NReqs;
  constexpr auto mpiReqBytes  = nPendingReqs * sizeof(MPI_Request);

  using req_buffer_t = SmallVector<MPI_Request, mpiReqBytes>;

  typename req_buffer_t::arena  reqs_arena{};
  typename req_buffer_t::vector reqs(
      nPendingReqs,
      MPI_REQUEST_NULL,
      typename req_buffer_t::allocator{reqs_arena});

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const totalReqs      = 2 * totalExchanges;
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  // allocate the communication state which provides the receive buffer enough
  // blocks of size blocksize.

  detail::CommState<value_type, NReqs> commState{std::size_t(blocksize)};

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs = 0, nrreqs = 0, sphase = 0, rphase = 0;

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&commState](auto /*peer*/, auto reqIdx) {
    return commState.receive_allocate(reqIdx);
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

  auto chunks_to_merge = std::vector<std::pair<InputIt, InputIt>>{};
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
      for (auto&& idx : fmpi::range(0, static_cast<int>(reqsInFlight))) {
        // mark everything as completed
        commState.receive_complete(idx);
      }

      auto const& completedChunks = commState.completed_receives();

      std::transform(
          std::begin(completedChunks),
          std::end(completedChunks),
          std::back_inserter(chunks_to_merge),
          [blocksize](auto b) { return std::make_pair(b, b + blocksize); });

      op(chunks_to_merge, out);
    }
    trace.tock(MERGE);
    trace.put(detail::N_COMM_ROUNDS, 1);
  }
  else {
    using index_type = int;
    using indices_buffer_t =
        SmallVector<index_type, nPendingReqs * sizeof(index_type)>;

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

    RTLX_ASSERT(indices.size() <= nPendingReqs);

    int count = 0;

    trace.tock(COMMUNICATION);

    while (ncReqs < totalReqs) {
      ++count;
      trace.tick(COMMUNICATION);

      auto* lastIdx = mpi::testsome(
          &(*reqs.begin()), &(*reqs.end()), &(*indices.begin()));

      auto const nCompleted = std::distance(&(*indices.begin()), lastIdx);

      FMPI_DBG(nCompleted);

      ncReqs += nCompleted;

      auto const reqsCompleted =
          std::make_pair(indices.begin(), indices.begin() + nCompleted);

      FMPI_DBG_RANGE(reqsCompleted.first, reqsCompleted.second);

      // we want all receive requests on the left, and send requests on the
      // right
      auto const fstSentIdx = std::partition(
          reqsCompleted.first,
          reqsCompleted.second,
          [reqsInFlight](auto req) {
            // left half of reqs array array are receives
            return req < static_cast<int>(reqsInFlight);
          });

      auto const nrecv = std::distance(reqsCompleted.first, fstSentIdx);

      auto const nsent = nCompleted - nrecv;

      FMPI_DBG_STREAM(
          me << " available chunks to merge: "
             << commState.completed_receives().size() + nrecv);

      // mark all receives
      std::for_each(
          reqsCompleted.first, fstSentIdx, [&commState](auto reqIdx) {
            commState.receive_complete(reqIdx);
          });

      nrecvTotal += nrecv;

      auto const nrecvOpen = totalExchanges - nrreqs;

      auto const nslotsRecv = std::min(
          std::min<std::size_t>(nrecv, nrecvOpen),
          commState.available_slots());

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

      auto const mergeCount =
          commState.completed_receives().size() + chunks_to_merge.size();

      // minimum number of chunks to merge: ideally we have a full level2
      // cache
      bool const enough_work = mergeCount >= utilization_threshold;

      FMPI_DBG_STREAM("ready chunks: " << mergeCount);

      if (enough_work || (allReceivesDone && mergeCount)) {
        // 1) copy completed chunks into std::vector (API requirements)

        trace.tick(MERGE);

        auto const& completedChunks = commState.completed_receives();

        std::transform(
            std::begin(completedChunks),
            std::end(completedChunks),
            std::back_inserter(chunks_to_merge),
            [blocksize](auto b) { return std::make_pair(b, b + blocksize); });

        FMPI_DBG_STREAM(
            me << " merging " << chunks_to_merge.size() << " chunks");

        // 2) merge all chunks
        op(chunks_to_merge, outIt);

        // 3) increase out iterator
        auto const nmerged = chunks_to_merge.size() * blocksize;

        outIt += nmerged;

        mergedChunksPsum.push_back(mergedChunksPsum.back() + nmerged);

        // 4) reset all completed Chunks
        commState.release_completed();
        chunks_to_merge.clear();

        trace.tock(MERGE);
      }
    }

    trace.tick(MERGE);
    trace.put(detail::N_COMM_ROUNDS, count);

    auto const needsFinalMerge = mergedChunksPsum.size() > 2;
    if (needsFinalMerge) {
      RTLX_ASSERT(chunks_to_merge.empty());

      using merge_buffer_t = tlx::
          SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

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
}

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
  using pointer    = typename std::iterator_traits<OutputIt>::pointer;

  auto me = ctx.rank();
  auto nr = ctx.size();

  std::ostringstream os;
  os << Schedule::NAME << "Waitall" << NReqs;
  auto trace = rtlx::TimeTrace{os.str()};

  auto const schedule = Schedule{};


  FMPI_DBG_STREAM("running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  std::array<MPI_Request, 2 * NReqs> reqs;
  reqs.fill(MPI_REQUEST_NULL);

  auto const totalExchanges =
      static_cast<std::size_t>(schedule.phaseCount(ctx));
  auto const reqsInFlight = std::min(totalExchanges - 1, NReqs);

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  using buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  std::vector<std::pair<pointer, pointer>> chunks;

  std::vector<std::size_t> mergePsum;

  chunks.reserve(reqsInFlight + 1);

  // We can save the local copy and do it with the merge operation itself
  chunks.push_back(std::make_pair(
      begin + me * blocksize, begin + me * blocksize + blocksize));

  auto const nrounds = tlx::div_ceil(totalExchanges, reqsInFlight);

  mergePsum.reserve(nrounds + 2);
  mergePsum.push_back(0);

  FMPI_DBG(nrounds);

  std::size_t rphase = 0, sphase = 0;
  auto        mergebuf = out;

  auto const nels   = nr * blocksize;
  auto const nbytes = nels * sizeof(value_type);
  buffer_t   buffer{nels};

  auto const allFitsInL2Cache = (nbytes < fmpi::CACHELEVEL2_SIZE);

  int32_t count = 0;
  for (auto&& win : range<std::size_t>(nrounds)) {
    ++count;
    trace.tick(COMMUNICATION);
    static_cast<void>(win);
    std::size_t nrreqs;

    for (nrreqs = 0; nrreqs < reqsInFlight && rphase < totalExchanges;
         ++rphase) {
      // receive from
      auto recvfrom = schedule.recvRank(ctx, rphase);

      if (recvfrom == me) continue;

      FMPI_DBG(recvfrom);

      auto* recvBuf = allFitsInL2Cache
                          ? std::next(buffer.begin(), recvfrom * blocksize)
                          : std::next(buffer.begin(), nrreqs * blocksize);

      FMPI_CHECK(mpi::irecv(
          recvBuf, blocksize, recvfrom, EXCH_TAG_RING, ctx, &reqs[nrreqs]));

      chunks.push_back(
          std::make_pair(recvBuf, std::next(recvBuf, blocksize)));

      ++nrreqs;
    }

    for (std::size_t nsreqs = 0;
         nsreqs < reqsInFlight && sphase < totalExchanges;
         ++sphase) {
      auto sendto = schedule.sendRank(ctx, sphase);

      if (sendto == me) continue;

      FMPI_DBG(sendto);

      FMPI_CHECK(mpi::isend(
          std::next(begin, sendto * blocksize),
          blocksize,
          sendto,
          EXCH_TAG_RING,
          ctx,
          &reqs[reqsInFlight + nsreqs++]));
    }

    FMPI_CHECK(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));
    trace.tock(COMMUNICATION);

    if (!allFitsInL2Cache && !chunks.empty()) {
      trace.tick(MERGE);
      op(chunks, mergebuf);
      auto const nMerged = chunks.size() * blocksize;
      mergebuf += nMerged;
      mergePsum.push_back(mergePsum.back() + nMerged);
      chunks.clear();
      trace.tock(MERGE);
    }

    FMPI_DBG_RANGE(out, mergebuf);
  }

  trace.tick(MERGE);


  auto mergeSrc = buffer.begin();
  auto target   = &*out;

  // if we have intermediate merge operations then we have already merged
  // chunks in the out buffer and use the dedicated merge buffer as destinated
  // merged buffer.
  // Otherwise, we merge directly into the out buffer.
  FMPI_DBG(mergePsum);
  if (!allFitsInL2Cache) {
    FMPI_DBG(mergePsum);
    std::swap(mergeSrc, target);
    std::transform(
        std::begin(mergePsum),
        std::prev(std::end(mergePsum)),
        std::next(std::begin(mergePsum)),
        std::back_inserter(chunks),
        [rbuf = mergeSrc](auto first, auto last) {
          return std::make_pair(
              std::next(rbuf, first), std::next(rbuf, last));
        });
  }

  FMPI_DBG("final merge");
  FMPI_DBG(chunks.size());

  op(chunks, target);

  if (target != &*out) {
    std::move(target, target + nels, out);
  }

  trace.tock(MERGE);
  trace.put(detail::N_COMM_ROUNDS, count);

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
}  // namespace fmpi
#endif
