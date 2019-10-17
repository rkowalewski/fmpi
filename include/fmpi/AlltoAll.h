#ifndef ALLTOALL_H
#define ALLTOALL_H

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stack>

#include <tlx/simple_vector.hpp>
#include <tlx/stack_allocator.hpp>

#include <rtlx/Assert.h>
#include <rtlx/Trace.h>

#include <fmpi/Schedule.h>
#include <fmpi/Debug.h>
#include <fmpi/Constants.h>
#include <fmpi/detail/CommState.h>

#include <fmpi/NumericRange.h>


// Other AllToAll Algorithms

namespace fmpi {

template <
    AllToAllAlgorithm algo,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void scatteredPairwiseWaitsome(
    InputIt                begin,
    OutputIt               out,
    int                    blocksize,
    mpi::MpiCommCtx const& ctx,
    Op&&                   op)
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
      utilization_threshold, "at least two concurrent receives required");

  using algo_type   = typename detail::selectAlgorithm<algo>::type;
  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::type_mapper<value_type>::type();

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::string s;

  if (rtlx::TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwiseWaitsome" << algo_type::NAME << NReqs;
    s = os.str();
  }

  auto trace = rtlx::TimeTrace{ctx.rank(), s};

  FMPI_DBG_STREAM("running algorithm " << s << ", blocksize: " << blocksize);

  trace.tick(COMMUNICATION);

  std::array<MPI_Request, 2 * NReqs> reqs{};
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const totalReqs      = 2 * totalExchanges;
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  // allocate the communication state which provides the receive buffer enough
  // blocks of size blocksize.

  detail::CommState<value_type, NReqs> commState{std::size_t(blocksize)};

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs = 0, nrreqs = 0, sphase = 0, rphase = 0;

  auto rschedule = [&ctx](auto phase) {
    algo_type commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&commState](auto /*peer*/, auto reqIdx) {
    return commState.receive_allocate(reqIdx);
  };

  auto receiveOp =
      [reqs = &reqs[0], blocksize, mpi_datatype, comm = ctx.mpiComm()](
          auto* buf, auto peer, auto reqIdx) {
        FMPI_DBG_STREAM("receiving from " << peer << " reqIdx " << reqIdx);

        return MPI_Irecv(
            buf,
            blocksize,
            mpi_datatype,
            peer,
            100,
            comm,
            std::next(reqs, reqIdx));
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
    algo_type commAlgo{};
    return commAlgo.sendRank(ctx, phase);
  };

  auto sbufAlloc = [begin, blocksize](auto peer, auto /*reqIdx*/) {
    return &*std::next(begin, peer * blocksize);
  };

  auto sendOp =
      [reqs = &reqs[0], blocksize, mpi_datatype, comm = ctx.mpiComm()](
          auto* buf, auto peer, auto reqIdx) {
        FMPI_DBG_STREAM("sending to " << peer << " reqIdx " << reqIdx);
        return MPI_Isend(
            buf,
            blocksize,
            mpi_datatype,
            peer,
            100,
            comm,
            std::next(reqs, reqIdx));
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
    RTLX_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    trace.tock(COMMUNICATION);

    trace.tick(MERGE);

    {
      auto range = fmpi::range(0, static_cast<int>(reqsInFlight));

      for (auto const& idx : range) {
        // mark everything as completed
        commState.receive_complete(idx);
      }

      auto& completedChunks = commState.completed_receives();

      std::copy(
          std::begin(completedChunks),
          std::end(completedChunks),
          std::back_inserter(chunks_to_merge));

      op(chunks_to_merge, out);
    }
    trace.tock(MERGE);
  }
  else {
    // receives are done but we still wait for send requests
    // Let's merge a deeper level in the tree
    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    size_t ncReqs = 0, ncrReqs = 0;
    auto   outIt = out;

    // request indexes indexes which have been completed so far
    std::array<int, reqs.size()> reqsCompleted{};

    std::vector<std::size_t> mergedChunksPsum;
    mergedChunksPsum.reserve(nr);
    mergedChunksPsum.push_back(0);

    trace.tock(COMMUNICATION);

    while (ncReqs < totalReqs) {
      int nReqCompleted;

      trace.tick(COMMUNICATION);

      RTLX_ASSERT_RETURNS(
          MPI_Waitsome(
              reqs.size(),
              &(reqs[0]),
              &nReqCompleted,
              &(reqsCompleted[0]),
              MPI_STATUSES_IGNORE),
          MPI_SUCCESS);

      RTLX_ASSERT(nReqCompleted != MPI_UNDEFINED);
      ncReqs += nReqCompleted;

      auto fstCompletedReq = std::begin(reqsCompleted);
      auto lstCompletedReq = fstCompletedReq + nReqCompleted;

      FMPI_DBG_RANGE(fstCompletedReq, lstCompletedReq);

      // reset all MPI Requests
      std::for_each(fstCompletedReq, lstCompletedReq, [&reqs](auto reqIdx) {
        reqs[reqIdx] = MPI_REQUEST_NULL;
      });

      // we want all receive requests on the left, and send requests on the
      // right
      auto const fstRecvRequest = fstCompletedReq;
      auto const fstSendRequest = std::partition(
          fstCompletedReq, lstCompletedReq, [reqsInFlight](auto req) {
            // left half of reqs array array are receives
            return req < static_cast<int>(reqsInFlight);
          });

      auto const nReceivedChunks =
          std::distance(fstRecvRequest, fstSendRequest);
      auto const nSentChunks = std::distance(fstSendRequest, lstCompletedReq);

      FMPI_DBG_STREAM(
          me << " available chunks to merge: "
             << commState.completed_receives().size() + nReceivedChunks);

      // mark all receives
      std::for_each(
          fstRecvRequest, fstSendRequest, [&commState](auto reqIdx) {
            commState.receive_complete(reqIdx);
          });

      ncrReqs += nReceivedChunks;

      RTLX_ASSERT((nReceivedChunks + nSentChunks) == nReqCompleted);

      // Number of open receives in this round.
      auto const maxVals = {// Number of total open receives
                            totalExchanges - nrreqs,
                            // Number of free receive slots
                            static_cast<size_t>(nReceivedChunks),
                            // number of free memory slots
                            commState.available_slots()};

      auto const maxRecvs =
          *std::min_element(std::begin(maxVals), std::end(maxVals));

      FMPI_DBG_STREAM("max receives in this round: " << maxRecvs);

      RTLX_ASSERT((fstRecvRequest + maxRecvs) <= fstSendRequest);

      rphase = detail::enqueueMpiOps(
          rphase,
          me,
          maxRecvs,
          rschedule,
          [fstRecvRequest](auto nreqs) {
            return *std::next(fstRecvRequest, nreqs);
          },
          rbufAlloc,
          receiveOp);

      nrreqs += maxRecvs;

      auto const maxSents =
          std::min<std::size_t>(totalExchanges - nsreqs, nSentChunks);

      sphase = detail::enqueueMpiOps(
          sphase,
          me,
          maxSents,
          sschedule,
          [fstSendRequest](auto nreqs) {
            return *std::next(fstSendRequest, nreqs);
          },
          sbufAlloc,
          sendOp);

      nsreqs += maxSents;

      trace.tock(COMMUNICATION);

      trace.tick(MERGE);

      auto const allReceivesDone = ncrReqs == totalExchanges;

      auto const mergeCount =
          commState.completed_receives().size() + chunks_to_merge.size();

      // minimum number of chunks to merge: ideally we have a full level2
      // cache
      bool const enough_merges_available =
          mergeCount >= utilization_threshold;

      FMPI_DBG_STREAM("ready chunks: " << mergeCount);

      if (enough_merges_available || (allReceivesDone && mergeCount)) {
        // 1) copy completed chunks into std::vector (API requirements)
        std::copy(
            std::begin(commState.completed_receives()),
            std::end(commState.completed_receives()),
            std::back_inserter(chunks_to_merge));

        FMPI_DBG_STREAM(
            me << " merging " << chunks_to_merge.size() << " chunks");

        // 2) merge all chunks
        op(chunks_to_merge, outIt);

        // 3) increase out iterator
        outIt += chunks_to_merge.size() * blocksize;
        mergedChunksPsum.push_back(
            mergedChunksPsum.back() + chunks_to_merge.size() * blocksize);

        // 4) reset all completed Chunks
        commState.release_completed();
        chunks_to_merge.clear();
      }
      else {
        auto const sentReqsOpen    = allReceivesDone && (ncReqs < totalReqs);
        auto const needsFinalMerge = mergedChunksPsum.size() > 2;

        if (sentReqsOpen && needsFinalMerge) {
          RTLX_ASSERT(chunks_to_merge.empty());

          auto mergeBuffer = merge_buffer_t{std::size_t(nr) * blocksize};

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

          op(chunks_to_merge, mergeBuffer.begin());
          chunks_to_merge.clear();
          mergedChunksPsum.erase(
              std::next(std::begin(mergedChunksPsum)),
              std::prev(std::end(mergedChunksPsum)));

          FMPI_DBG(mergedChunksPsum);

          std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
        }
      }
      trace.tock(MERGE);
    }
    trace.tick(MERGE);
    if (mergedChunksPsum.size() > 2) {
      RTLX_ASSERT(chunks_to_merge.empty());

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
  RTLX_ASSERT(std::is_sorted(out, out + nr * blocksize));
}

template <AllToAllAlgorithm algo, class InputIt, class OutputIt, class Op>
inline void scatteredPairwise(
    InputIt                begin,
    OutputIt               out,
    int                    blocksize,
    mpi::MpiCommCtx const& ctx,
    Op&&                   op)
{
  auto nr = ctx.size();
  auto me = ctx.rank();

  using algo_type  = typename detail::selectAlgorithm<algo>::type;
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto mpi_datatype = mpi::type_mapper<value_type>::type();
  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  std::string s;

  if (rtlx::TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwise" << algo_type::NAME;
    s = os.str();
  }

  auto trace = rtlx::TimeTrace{me, s};

  trace.tick(COMMUNICATION);
  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      &rbuf[0] + me * blocksize);

  auto commAlgo = algo_type{};

  for (int r = 0; r < nr; ++r) {
    auto sendto   = commAlgo.sendRank(ctx, r);
    auto recvfrom = commAlgo.recvRank(ctx, r);

    if (sendto == me) {
      sendto = MPI_PROC_NULL;
    }
    if (recvfrom == me) {
      recvfrom = MPI_PROC_NULL;
    }
    if (sendto == MPI_PROC_NULL && recvfrom == MPI_PROC_NULL) {
      continue;
    }

    FMPI_DBG(sendto);
    FMPI_DBG(recvfrom);

    RTLX_ASSERT_RETURNS(
        MPI_Sendrecv(
            std::next(begin, sendto * blocksize),
            static_cast<int>(blocksize),
            mpi_datatype,
            sendto,
            100,
            std::next(&(rbuf[0]), recvfrom * blocksize),
            static_cast<int>(blocksize),
            mpi_datatype,
            recvfrom,
            100,
            ctx.mpiComm(),
            MPI_STATUS_IGNORE),
        MPI_SUCCESS);
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range(0, nr * blocksize, blocksize);

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
  RTLX_ASSERT(std::is_sorted(out, out + nr * blocksize));
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt                begin,
    OutputIt               out,
    int                    blocksize,
    mpi::MpiCommCtx const& ctx,
    Op&&                   op)
{
  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::type_mapper<value_type>::type();

  auto nr = ctx.size();
  auto me = ctx.rank();

  auto trace = rtlx::TimeTrace{me, "AlltoAll"};

  trace.tick(COMMUNICATION);

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  RTLX_ASSERT_RETURNS(
      MPI_Alltoall(
          std::addressof(*begin),
          blocksize,
          mpi_datatype,
          &rbuf[0],
          blocksize,
          mpi_datatype,
          ctx.mpiComm()),
      MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range(0, nr * blocksize, blocksize);

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

  RTLX_ASSERT(std::is_sorted(out, out + nr * blocksize));
}

#if 0
template <class InputIt, class OutputIt, class Op, size_t NReqs = 1>
inline void scatteredPairwiseWaitany(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  std::string s;
  if (rtlx::TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwiseWaitany" << NReqs;
    s = os.str();
  }

  auto trace = rtlx::TimeTrace{me, s};

  trace.tick(COMMUNICATION);

  std::array<MPI_Request, 2 * NReqs> reqs;
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  // local copy
  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const totalReqs      = 2 * totalExchanges;
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs, nrreqs;
  for (nrreqs = 0; nrreqs < reqsInFlight; ++nrreqs) {
    // receive from
    auto recvfrom = mod(me - static_cast<int>(nrreqs) - 1, nr);

    P(me << " recvfrom block " << recvfrom << ", req " << nrreqs);
    // post request
    RTLX_ASSERT_RETURNS(
        MPI_Irecv(
            std::next(out, recvfrom * blocksize),
            blocksize,
            mpi_datatype,
            recvfrom,
            100,
            comm,
            &(reqs[nrreqs])),
        MPI_SUCCESS);
  }

  for (nsreqs = 0; nsreqs < reqsInFlight; ++nsreqs) {
    // receive from
    auto sendto = mod(me + static_cast<int>(nsreqs) + 1, nr);

    auto reqIdx = nsreqs + reqsInFlight;

    P(me << " sendto block " << sendto << ", req " << reqIdx);
    RTLX_ASSERT_RETURNS(
        MPI_Isend(
            std::next(begin, sendto * blocksize),
            blocksize,
            mpi_datatype,
            sendto,
            100,
            comm,
            &(reqs[reqIdx])),
        MPI_SUCCESS);
  }

  P(me << " total reqs " << totalReqs);
  if (reqsInFlight == totalExchanges) {
    // We are already done
    // Wait for previous round
    RTLX_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);
  }
  else {
    size_t ncReqs = 0;

    while (ncReqs < totalReqs) {
      int reqCompleted;

      RTLX_ASSERT_RETURNS(
          MPI_Waitany(
              reqs.size(), &(reqs[0]), &reqCompleted, MPI_STATUS_IGNORE),
          MPI_SUCCESS);

      P(me << " completed req " << reqCompleted);

      reqs[reqCompleted] = MPI_REQUEST_NULL;
      ++ncReqs;
      P(me << " ncReqs " << ncReqs);
      RTLX_ASSERT(reqCompleted >= 0);
      if (reqCompleted < static_cast<int>(reqsInFlight)) {
        // a receive request is done, so post a new one...

        // but we really need to check if we really need to perform another
        // request, because a MPI_Request_NULL could be completed as well
        if (nrreqs < totalExchanges) {
          // receive from
          auto recvfrom = mod(me - static_cast<int>(nrreqs) - 1, nr);

          P(me << " recvfrom " << recvfrom);

          RTLX_ASSERT_RETURNS(
              MPI_Irecv(
                  std::next(out, recvfrom * blocksize),
                  blocksize,
                  mpi_datatype,
                  recvfrom,
                  100,
                  comm,
                  &(reqs[reqCompleted])),
              MPI_SUCCESS);
          ++nrreqs;
        }
      }
      else {
        // a send request is done, so post a new one...
        if (nsreqs < totalExchanges) {
          // receive from
          auto sendto = mod(me + static_cast<int>(nsreqs) + 1, nr);

          P(me << " sendto " << sendto);

          RTLX_ASSERT_RETURNS(
              MPI_Isend(
                  std::next(begin, sendto * blocksize),
                  blocksize,
                  mpi_datatype,
                  sendto,
                  100,
                  comm,
                  &(reqs[reqCompleted])),
              MPI_SUCCESS);
          ++nsreqs;
        }
      }
    }
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);
#if 0
  std::vector<std::pair<OutputIt, OutputIt>> seqs;
  seqs.reserve(nr);

  for (size_t i = 0; i < std::size_t(nr); ++i) {
    seqs.push_back(
        std::make_pair(out + i * blocksize, out + (i + 1) * blocksize));
  }

  auto merge_buf =
      std::unique_ptr<value_type[]>(new value_type[blocksize * nr]);

  __gnu_parallel::multiway_merge(
      seqs.begin(),
      seqs.end(),
      merge_buf.get(),
      blocksize * nr,
      std::less<value_type>{},
      __gnu_parallel::sequential_tag{});

  std::copy(merge_buf.get(), merge_buf.get() + blocksize * nr, out);

#endif
  trace.tock(MERGE);
}
#endif
}  // namespace fmpi
#endif
