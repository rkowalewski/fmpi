#ifndef ALLTOALL_H
#define ALLTOALL_H

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stack>

#include <parallel/algorithm>

#include <tlx/simple_vector.hpp>
#include <tlx/stack_allocator.hpp>

// Other AllToAll Algorithms
#include <Bruck.h>
#include <Constants.h>
#include <Debug.h>
#include <Factor.h>
#include <Mpi.h>
#include <Trace.h>

namespace a2a {
#if 0

template <class InputIt, class OutputIt, class Op>
inline void flatHandshake(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  auto trace = TimeTrace{me, "FlatHandshake"};

  trace.tick(COMMUNICATION);

  auto sendRecvPair = [me, nr](auto step) {
    return std::make_pair(mod(me + step, nr), mod(me - step, nr));
  };

  int sendto, recvfrom;

  std::tie(sendto, recvfrom) = sendRecvPair(1);

  auto reqs = a2a::sendrecv(
      std::next(begin, sendto * blocksize),
      blocksize,
      sendto,
      100,
      std::next(out, recvfrom * blocksize),
      blocksize,
      recvfrom,
      100,
      comm);

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  for (int i = 2; i < nr; ++i) {
    std::tie(sendto, recvfrom) = sendRecvPair(i);

    // Wait for previous round
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    reqs = a2a::sendrecv(
        std::next(begin, sendto * blocksize),
        blocksize,
        sendto,
        100,
        std::next(out, recvfrom * blocksize),
        blocksize,
        recvfrom,
        100,
        comm);
  }

  // Wait for previous round
  A2A_ASSERT_RETURNS(
      MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE), MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  trace.tock(MERGE);
}

template <class InputIt, class OutputIt, class Op, size_t NReqs = 1>
inline void scatteredPairwise(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  std::string s = "";
  if (TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwise" << NReqs;
    s = os.str();
  }
  auto trace = TimeTrace{me, s};

  trace.tick(COMMUNICATION);

  std::array<MPI_Request, 2 * NReqs> reqs;
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

#if 0
  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);
#endif

  for (int ii = 0; ii < nr; ii += NReqs) {
    auto ss = std::min<int>(nr - ii, NReqs);

    P("ii block: " << ss << ", ss: " << ss);
    for (auto i = 0; i < ss; ++i) {
      // Overlapping first round...
      auto recvfrom = mod(me - i - ii, nr);

      P(me << " recvfrom " << recvfrom);

      A2A_ASSERT_RETURNS(
          MPI_Irecv(
              std::next(out, recvfrom * blocksize),
              blocksize,
              mpi_datatype,
              recvfrom,
              100,
              comm,
              &(reqs[i])),
          MPI_SUCCESS);
    }

    for (auto i = 0; i < ss; ++i) {
      // Overlapping first round...
      auto sendto = mod(me + i + ii, nr);

      P(me << " sendto " << sendto);

      A2A_ASSERT_RETURNS(
          MPI_Isend(
              std::next(begin, sendto * blocksize),
              blocksize,
              mpi_datatype,
              sendto,
              100,
              comm,
              &(reqs[i + ss])),
          MPI_SUCCESS);
    }

    // Wait for previous round
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);
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
  if (TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwiseWaitany" << NReqs;
    s = os.str();
  }

  auto trace = TimeTrace{me, s};

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

  A2A_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs, nrreqs;
  for (nrreqs = 0; nrreqs < reqsInFlight; ++nrreqs) {
    // receive from
    auto recvfrom = mod(me - static_cast<int>(nrreqs) - 1, nr);

    P(me << " recvfrom block " << recvfrom << ", req " << nrreqs);
    // post request
    A2A_ASSERT_RETURNS(
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
    A2A_ASSERT_RETURNS(
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
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);
  }
  else {
    size_t ncReqs = 0;

    while (ncReqs < totalReqs) {
      int reqCompleted;

      A2A_ASSERT_RETURNS(
          MPI_Waitany(
              reqs.size(), &(reqs[0]), &reqCompleted, MPI_STATUS_IGNORE),
          MPI_SUCCESS);

      P(me << " completed req " << reqCompleted);

      reqs[reqCompleted] = MPI_REQUEST_NULL;
      ++ncReqs;
      P(me << " ncReqs " << ncReqs);
      A2A_ASSERT(reqCompleted >= 0);
      if (reqCompleted < static_cast<int>(reqsInFlight)) {
        // a receive request is done, so post a new one...

        // but we really need to check if we really need to perform another
        // request, because a MPI_Request_NULL could be completed as well
        if (nrreqs < totalExchanges) {
          // receive from
          auto recvfrom = mod(me - static_cast<int>(nrreqs) - 1, nr);

          P(me << " recvfrom " << recvfrom);

          A2A_ASSERT_RETURNS(
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

          A2A_ASSERT_RETURNS(
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

namespace detail {
template <class T, std::size_t NReqs>
class CommState {
  using buffer_t =
      tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using size_type  = typename buffer_t::size_type;
  using iterator_t = typename buffer_t::iterator;
  using chunk_t    = std::pair<iterator_t, iterator_t>;

  // Buffer for up to 2 * NReqs pending chunks
  static constexpr size_t MAX_FREE_CHUNKS      = 2 * NReqs;
  static constexpr size_t MAX_COMPLETED_CHUNKS = MAX_FREE_CHUNKS;

  template <std::size_t N>
  using chunks_storage_t =
      std::vector<chunk_t, tlx::StackAllocator<chunk_t, N * sizeof(chunk_t)>>;

 public:
  CommState() = default;

  explicit CommState(size_type blocksize)
    : m_completed(
          0,
          chunk_t{},
          typename chunks_storage_t<MAX_COMPLETED_CHUNKS>::allocator_type{
              m_arena_completed})
    , m_freelist(chunks_storage_t<MAX_FREE_CHUNKS>{
          0,
          chunk_t{},
          typename chunks_storage_t<MAX_FREE_CHUNKS>::allocator_type{
              m_arena_freelist}})
    , m_buffer(blocksize * MAX_FREE_CHUNKS)
  {
    std::fill(std::begin(m_pending), std::end(m_pending), chunk_t{});

    auto r = a2a::range<int>(MAX_FREE_CHUNKS - 1, -1, -1);

    for (auto const& block : r) {
      auto f = std::next(std::begin(m_buffer), block * blocksize);
      auto l = std::next(f, blocksize);
      m_freelist.push(std::make_pair(f, l));
    }
  }

  iterator_t receive_allocate(int key)
  {
    A2A_ASSERT(m_arena_freelist.size() > 0);
    A2A_ASSERT(0 <= key && std::size_t(key) < NReqs);
    A2A_ASSERT(!m_freelist.empty() && m_freelist.size() <= MAX_FREE_CHUNKS);

    // access last block remove it from stack
    auto freeBlock = m_freelist.top();
    m_freelist.pop();

    m_pending[key] = freeBlock;
    return freeBlock.first;
  }

  void receive_complete(int key)
  {
    A2A_ASSERT(m_arena_completed.size() > 0);
    A2A_ASSERT(0 <= key && std::size_t(key) < NReqs);
    A2A_ASSERT(m_completed.size() < MAX_COMPLETED_CHUNKS);

    auto block = m_pending[key];
    m_completed.push_back(block);
  }

  void release_completed()
  {
    A2A_ASSERT(m_arena_freelist.size() > 0);
    A2A_ASSERT(m_completed.size() <= MAX_COMPLETED_CHUNKS);

    for (auto const& completed : m_completed) {
      m_freelist.push(completed);
    }

    // 2) reset completed receives
    m_completed.clear();

    A2A_ASSERT(!m_freelist.empty() && m_freelist.size() <= MAX_FREE_CHUNKS);
  }

  std::size_t available_slots() const noexcept
  {
    return m_freelist.size();
  }

  chunks_storage_t<MAX_COMPLETED_CHUNKS> const& completed_receives() const
      noexcept
  {
    return m_completed;
  }

 private:
  tlx::StackArena<MAX_COMPLETED_CHUNKS * sizeof(chunk_t)> m_arena_completed{};
  chunks_storage_t<MAX_COMPLETED_CHUNKS>                  m_completed{};

  tlx::StackArena<MAX_FREE_CHUNKS * sizeof(chunk_t)>     m_arena_freelist{};
  std::stack<chunk_t, chunks_storage_t<MAX_FREE_CHUNKS>> m_freelist{};

  std::array<chunk_t, NReqs> m_pending{};

  buffer_t m_buffer{};
};
}  // namespace detail

template <class InputIt, class OutputIt, class Op, size_t NReqs = 2>
inline void scatteredPairwiseWaitsome(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
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

  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  std::string s;
  if (TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "ScatteredPairwiseWaitany" << NReqs;
    s = os.str();
  }

  auto trace = TimeTrace{me, s};

  trace.tick(COMMUNICATION);

  std::array<MPI_Request, 2 * NReqs> reqs{};
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const totalReqs      = 2 * totalExchanges;
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  auto recvRank = [nr, me](int phase) {
    if (a2a::isPow2(static_cast<unsigned>(nr))) {
      return me ^ phase;
    }
    return mod(me - phase, nr);
  };

  auto sendRank = [nr, me](int phase) {
    if (a2a::isPow2(static_cast<unsigned>(nr))) {
      return me ^ phase;
    }

    return mod(me + phase, nr);
  };

  // allocate the communication state which provides the receive buffer enough
  // blocks of size blocksize.
  detail::CommState<value_type, NReqs> commState{std::size_t(blocksize)};

  A2A_ASSERT(2 * reqsInFlight <= reqs.size());

  std::size_t nsreqs, nrreqs;

  for (nrreqs = 0; nrreqs < reqsInFlight; ++nrreqs) {
    // receive from
    auto recvfrom = recvRank(nrreqs + 1);

    P(me << " recvfrom block " << recvfrom << ", req " << nrreqs);

    auto* rbuf = commState.receive_allocate(nrreqs);

    A2A_ASSERT(rbuf);

    // post request
    A2A_ASSERT_RETURNS(
        MPI_Irecv(
            rbuf,
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
    auto sendto = sendRank(nsreqs + 1);

    auto reqIdx = nsreqs + reqsInFlight;

    P(me << " sendto block " << sendto << ", req " << reqIdx);
    A2A_ASSERT_RETURNS(
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
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    trace.tock(COMMUNICATION);

    trace.tick(MERGE);

    {
      auto range = a2a::range<std::size_t>(0, reqsInFlight);

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

      A2A_ASSERT_RETURNS(
          MPI_Waitsome(
              reqs.size(),
              &(reqs[0]),
              &nReqCompleted,
              &(reqsCompleted[0]),
              MPI_STATUSES_IGNORE),
          MPI_SUCCESS);

      A2A_ASSERT(nReqCompleted != MPI_UNDEFINED);
      ncReqs += nReqCompleted;

      auto fstCompletedReq = std::begin(reqsCompleted);
      auto lstCompletedReq = fstCompletedReq + nReqCompleted;

      P(me << " completed reqs: "
           << tokenizeRange(fstCompletedReq, lstCompletedReq));

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

      P(me << " available chunks to merge: "
           << commState.completed_receives().size() + nReceivedChunks);

      // mark all receives
      std::for_each(
          fstRecvRequest, fstSendRequest, [&commState](auto reqIdx) {
            commState.receive_complete(reqIdx);
          });

      ncrReqs += nReceivedChunks;

      A2A_ASSERT((nReceivedChunks + nSentChunks) == nReqCompleted);

      // Number of open receives in this round.
      auto const maxVals = {// Number of total open receives
                            totalExchanges - nrreqs,
                            // Number of free receive slots
                            static_cast<size_t>(nReceivedChunks),
                            // number of free memory slots
                            commState.available_slots()};

      P(me << " open receives: " << totalExchanges - nrreqs
           << ", received chunks: " << nReceivedChunks
           << ", avaiable slots: " << commState.available_slots());
      auto const maxRecvs =
          *std::min_element(std::begin(maxVals), std::end(maxVals));

      P(me << " max receives: " << maxRecvs);
      A2A_ASSERT((fstRecvRequest + maxRecvs) <= fstSendRequest);
      for (auto it = fstRecvRequest; it < fstRecvRequest + maxRecvs; ++it) {
        auto recvfrom = recvRank(nrreqs + 1);

        auto reqIdx = *it;

        P(me << " allocating receive index: " << reqIdx);

        auto* buf = commState.receive_allocate(reqIdx);

        A2A_ASSERT(buf);

        P(me << " recvfrom " << recvfrom);

        A2A_ASSERT_RETURNS(
            MPI_Irecv(
                buf,
                blocksize,
                mpi_datatype,
                recvfrom,
                100,
                comm,
                &(reqs[reqIdx])),
            MPI_SUCCESS);
        ++nrreqs;
      }

      auto const maxSents =
          std::min<std::size_t>(totalExchanges - nsreqs, nSentChunks);

      for (auto it = fstSendRequest; it < fstSendRequest + maxSents; ++it) {
        auto reqIdx = *it;

        // a send request is done, so post a new one...
        // receive from
        auto sendto = sendRank(nsreqs + 1);

        P(me << " sendto " << sendto);

        A2A_ASSERT_RETURNS(
            MPI_Isend(
                std::next(begin, sendto * blocksize),
                blocksize,
                mpi_datatype,
                sendto,
                100,
                comm,
                &(reqs[reqIdx])),
            MPI_SUCCESS);
        ++nsreqs;
      }

      trace.tock(COMMUNICATION);

      trace.tick(MERGE);

      auto const allReceivesDone = ncrReqs == totalExchanges;

      auto const mergeCount =
          commState.completed_receives().size() + chunks_to_merge.size();
      // minimum number of chunks to merge: ideally we have a full level2
      // cache

      bool const enough_merges_available =
          mergeCount >= utilization_threshold;

      if (enough_merges_available || (allReceivesDone && mergeCount)) {
        // 1) copy completed chunks into another vector
        std::copy(
            std::begin(commState.completed_receives()),
            std::end(commState.completed_receives()),
            std::back_inserter(chunks_to_merge));

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
          // receives are done but we still wait for send requests
          // Let's merge a deeper level in the tree
          using buffer_t = tlx::SimpleVector<
              value_type,
              tlx::SimpleVectorMode::NoInitNoDestroy>;

          A2A_ASSERT(chunks_to_merge.empty());

          buffer_t mergeBuffer{std::size_t(nr) * blocksize};
          std::transform(
              std::begin(mergedChunksPsum),
              std::prev(std::end(mergedChunksPsum)),
              std::next(std::begin(mergedChunksPsum)),
              std::back_inserter(chunks_to_merge),
              [rbuf = out](auto first, auto last) {
                return std::make_pair(
                    std::next(rbuf, first), std::next(rbuf, last));
              });
          op(chunks_to_merge, mergeBuffer.begin());
          mergedChunksPsum.erase(
              std::next(std::begin(mergedChunksPsum)),
              std::prev(std::end(mergedChunksPsum)));
          std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
        }
      }
      trace.tock(MERGE);
    }
    trace.tick(MERGE);
    if (mergedChunksPsum.size() > 2) {
      // receives are done but we still wait for send requests
      // Let's merge a deeper level in the tree
      using buffer_t = tlx::
          SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

      A2A_ASSERT(chunks_to_merge.empty());

      buffer_t mergeBuffer{std::size_t(nr) * blocksize};
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
      // merge
      op(chunks_to_merge, mergeBuffer.begin());
      // switch buffer back to output iterator
      std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
    }

    trace.tock(MERGE);

    // final merge here
    P(me << " final merge: "
         << tokenizeRange(
                std::begin(mergedChunksPsum), std::end(mergedChunksPsum)));
  }
  A2A_ASSERT(std::is_sorted(out, out + nr * blocksize));
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
{
  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  int nr, me;
  MPI_Comm_size(comm, &nr);
  MPI_Comm_rank(comm, &me);

  auto trace = TimeTrace{me, "AlltoAll"};

  trace.tick(COMMUNICATION);

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  A2A_ASSERT_RETURNS(
      MPI_Alltoall(
          std::addressof(*begin),
          blocksize,
          mpi_datatype,
          &rbuf[0],
          blocksize,
          mpi_datatype,
          comm),
      MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = a2a::range(0, nr * blocksize, blocksize);

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
}  // namespace a2a
#endif
