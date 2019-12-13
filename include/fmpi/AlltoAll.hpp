#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <algorithm>
#include <cmath>
#include <fmpi/Constants.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/Memory.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/detail/CommState.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Request.hpp>
#include <future>
#include <memory>
#include <numeric>
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

template <typename F, typename... Ts>
inline auto make_async(F&& f, Ts&&... params)
{
  // Suggested in effective modern C++ to get true asynchrony
  return std::async(
      std::launch::async, std::forward<F>(f), std::forward<Ts>(params)...);
}

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
}

template <class T, std::size_t Capacity>
using lfq_fifo =
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>>;

template <class Iterator, std::size_t Capacity>
void push_fifo(
    Iterator begin,
    Iterator end,
    lfq_fifo<typename std::iterator_traits<Iterator>::value_type, Capacity>&
        fifo)
{
  FMPI_DBG_STREAM(
      "pushing " << std::distance(begin, end) << " on fifo of capacity "
                 << Capacity);
  RTLX_ASSERT(
      std::distance(begin, end) <=
      static_cast<typename std::iterator_traits<Iterator>::difference_type>(
          Capacity));
  for (auto it = begin; it != end;) {
    it = fifo.push(it, end);
  }
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

  constexpr auto winsz   = NReqs;
  constexpr auto winreqs = 2 * NReqs;

  using req_buffer_t = SmallVector<MPI_Request, winreqs>;

  typename req_buffer_t::arena  reqs_arena{};
  typename req_buffer_t::vector reqs(
      winreqs,
      MPI_REQUEST_NULL,
      typename req_buffer_t::allocator{reqs_arena});

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  using window_buffer =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto winbuf = window_buffer{blocksize * 2 * reqsInFlight};
  using bufit = typename window_buffer::iterator;
  using chunk = std::pair<bufit, bufit>;

  auto lfq_done = detail::lfq_fifo<chunk, winsz>{};

  auto lfq_freelist = detail::lfq_fifo<chunk, winreqs>{};
  {
    auto chunks = std::array<chunk, winreqs>{};
    auto r      = range<std::size_t>(chunks.size());

    std::transform(
        std::begin(r),
        std::end(r),
        std::begin(chunks),
        [begin = std::begin(winbuf), blocksize](auto idx) {
          return std::make_pair(
              std::next(begin, idx * blocksize),
              std::next(begin, (idx + 1) * blocksize));
        });

    detail::push_fifo(std::begin(chunks), std::end(chunks), lfq_freelist);
  }

  auto const lchunk = std::make_pair(
      std::next(begin, me * blocksize),
      std::next(begin, (me + 1) * blocksize));

  auto compute = detail::make_async([&lfq_done,
                                     &lfq_freelist,
                                     lchunk,
                                     n_messages = totalExchanges,
                                     nr,
                                     outputIt = out,
                                     op]() {
    // chunks to merge
    std::vector<chunk> chunks_to_merge;
    chunks_to_merge.reserve(n_messages);
    chunks_to_merge.push_back(lchunk);

    auto const blocksize = std::distance(lchunk.first, lchunk.second);

    // prefix sum over all processed chunks
    std::vector<chunk> processed;
    processed.reserve(nr);

    std::size_t n_arrivals = 0;

    auto first = outputIt;

    FMPI_DBG(n_messages);

    while (n_arrivals < n_messages) {
      auto const n_old = chunks_to_merge.size();
      auto const n_new = lfq_done.pop(std::back_inserter(chunks_to_merge));
      FMPI_DBG(n_new);
      auto const n_merges = n_old + n_new;

      n_arrivals += n_new;
      FMPI_DBG(n_arrivals);

      // minimum number of chunks to merge: ideally we have a full level2
      // cache
      bool const enough_work = n_merges >= utilization_threshold;

      FMPI_DBG(n_merges);

      if (enough_work) {
        // 2) merge all chunks
        auto last = op(chunks_to_merge, first);

        // 4) release completed buffers for future receives
        auto fbuf = std::begin(chunks_to_merge);

        if (first == outputIt) {
          std::advance(fbuf, 1);
        }

        detail::push_fifo(fbuf, std::end(chunks_to_merge), lfq_freelist);

        chunks_to_merge.clear();

        // 3) increase out iterator
        processed.emplace_back(std::make_pair(first, last));
        std::swap(first, last);
      }
      else {
        // we can eventually replace this with a wait method
        std::this_thread::yield();
      }
    }

    auto const nels = static_cast<std::size_t>(nr) * blocksize;

    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto mergeBuffer = merge_buffer_t{nels};
    // generate pairs of chunks to merge
    std::copy(
        std::begin(processed),
        std::end(processed),
        std::back_inserter(chunks_to_merge));

    FMPI_DBG(chunks_to_merge.size());
    // merge
    auto const last = op(chunks_to_merge, mergeBuffer.begin());

    RTLX_ASSERT(last == mergeBuffer.end());

    FMPI_DBG(processed);

    return std::move(mergeBuffer.begin(), mergeBuffer.end(), outputIt);
  });

  // allocate the communication state which provides the receive buffer
  // enough blocks of size blocksize.
  detail::CommState<value_type, NReqs> commState{};

  RTLX_ASSERT(2 * reqsInFlight <= reqs.size());

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&commState, &lfq_freelist](auto /*peer*/, auto reqIdx) {
    chunk c;
    while (!lfq_freelist.pop(c)) {
      // std::this_thread::yield();
    }
    commState.markOccupied(reqIdx, c);

    return c.first;
  };

  auto receiveOp = [&reqs, blocksize, &ctx](
                       auto* buf, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("receiving from " << peer << " reqIdx " << reqIdx);

    return mpi::irecv(
        buf, blocksize, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

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

  using index_type       = int;
  using indices_buffer_t = SmallVector<index_type, winsz>;

  typename indices_buffer_t::arena  indices_arena{};
  typename indices_buffer_t::vector indices(
      typename indices_buffer_t::allocator{indices_arena});

  indices.resize(std::distance(reqs.begin(), reqs.end()));
  // initially we can use the full array of request indices for send and
  // receives
  std::iota(indices.begin(), std::next(indices.begin(), reqsInFlight * 2), 0);

  std::vector<chunk> arrived_chunks;
  arrived_chunks.reserve(winsz);

  std::size_t const total_reqs = 2 * totalExchanges;
  std::size_t       nc_reqs    = 0;
  std::size_t       nsreqs     = 0;
  std::size_t       nrreqs     = 0;
  std::size_t       sphase     = 0;
  std::size_t       rphase     = 0;
  std::size_t       nSlotsRecv = reqsInFlight;
  std::size_t       nSlotsSend = reqsInFlight;

  // Yes, this is an int due to API requirements in our trace module
  int n_comm_rounds = 0;

  // Indices smaller than the pivot are receive requests, then we have send
  // requests
  auto sreqs_pivot = std::next(std::begin(indices), reqsInFlight);

  do {
    ++n_comm_rounds;

    FMPI_DBG(n_comm_rounds);

    FMPI_DBG("receiving...");
    FMPI_DBG(nSlotsRecv);

    rphase = detail::enqueueMpiOps(
        rphase,
        me,
        nSlotsRecv,
        rschedule,
        [it = std::begin(indices)](auto nreqs) {
          return *std::next(it, nreqs);
        },
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

    FMPI_ASSERT(
        (!(nc_reqs < total_reqs)) ||
        static_cast<std::size_t>(std::count(
            reqs.begin(), reqs.end(), MPI_REQUEST_NULL)) < reqs.size());

    auto* lastIdx =
        mpi::waitsome(&(*reqs.begin()), &(*reqs.end()), &(*indices.begin()));

    RTLX_ASSERT(lastIdx >= &*indices.begin());

    auto const nCompleted = std::distance(&(*indices.begin()), lastIdx);

    FMPI_DBG(nCompleted);

    nc_reqs += nCompleted;

    auto const reqsCompleted =
        std::make_pair(indices.begin(), indices.begin() + nCompleted);

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

    auto const nrecv = std::distance(reqsCompleted.first, sreqs_pivot);

    FMPI_DBG(nrecv);

    auto const nsent = nCompleted - nrecv;

    FMPI_DBG(nsent);

    FMPI_DBG_STREAM("pushing " << nrecv << " on queue...");

    {
      std::transform(
          reqsCompleted.first,
          sreqs_pivot,
          std::back_inserter(arrived_chunks),
          [&commState](auto reqIdx) {
            return commState.retrieveOccupied(reqIdx);
          });

      detail::push_fifo(
          std::begin(arrived_chunks), std::end(arrived_chunks), lfq_done);

      arrived_chunks.clear();
    }

    auto const nrecvOpen   = totalExchanges - nrreqs;
    auto const avail_slots = lfq_freelist.read_available();
    FMPI_DBG(avail_slots);

    nSlotsRecv =
        std::min(std::min<std::size_t>(nrecv, nrecvOpen), avail_slots);

    nSlotsSend = std::min<std::size_t>(totalExchanges - nsreqs, nsent);

    RTLX_ASSERT((reqsCompleted.first + nSlotsRecv) <= sreqs_pivot);

  } while (nc_reqs < total_reqs);

  trace.tock(COMMUNICATION);

  trace.put(detail::N_COMM_ROUNDS, n_comm_rounds);

  FMPI_DBG("waiting for compute");

  trace.tick(MERGE);
  auto last = compute.get();
  trace.tock(MERGE);
  RTLX_ASSERT(last == out + (std::size_t(nr) * blocksize));

}  // namespace fmpi

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
    auto const phaseCount = schedule.phaseCount(ctx);
    auto const winsize    = reqWin.winsize();

    std::size_t rphase;
    std::size_t sphase;
    std::tie(rphase, sphase) = initialPhases;

    FMPI_DBG(initialPhases);

    for (std::size_t nrreqs = 0; nrreqs < winsize && rphase < phaseCount;
         ++rphase) {
      // receive from
      auto recvfrom = schedule.recvRank(ctx, rphase);

      if (recvfrom == ctx.rank()) {
        continue;
      }

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

      if (sendto == ctx.rank()) {
        continue;
      }

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

    trace.tick(MERGE);
    op(reqWin.ready_pieces(), mergebuf);
    auto const nMerged = reqWin.ready_pieces().size() * blocksize;
    mergebuf += nMerged;
    mergePsum.push_back(mergePsum.back() + nMerged);
    reqWin.ready_pieces().clear();
    trace.tock(MERGE);

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

  FMPI_DBG(reqWin.ready_pieces());

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
