#ifndef BRUCK_H__INCLUDED
#define BRUCK_H__INCLUDED

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <parallel/algorithm>

// Other AllToAll Algorithms
#include <Bruck.h>
#include <Constants.h>
#include <Debug.h>
#include <Factor.h>
#include <Mpi.h>
#include <Trace.h>

namespace a2a {

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

template <class InputIt, class OutputIt, class Op, size_t NReqs = 1>
inline void scatteredPairwiseWaitany(
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

template <class InputIt, class OutputIt, class Op, size_t NReqs = 1>
inline void scatteredPairwiseWaitsome(
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

    std::array<int, reqs.size()> reqsCompleted;

    while (ncReqs < totalReqs) {
      int nReqCompleted;

      A2A_ASSERT_RETURNS(
          MPI_Waitsome(
              reqs.size(),
              &(reqs[0]),
              &nReqCompleted,
              &(reqsCompleted[0]),
              MPI_STATUS_IGNORE),
          MPI_SUCCESS);

      A2A_ASSERT(nReqCompleted != MPI_UNDEFINED);

      P(me << " completed req " << nReqCompleted);

      for (auto it = std::begin(reqsCompleted);
           it < std::begin(reqsCompleted) + nReqCompleted;
           ++it) {
        reqs[*it] = MPI_REQUEST_NULL;
      }

      ncReqs += nReqCompleted;
      P(me << " ncReqs " << ncReqs);
      for (auto it = std::begin(reqsCompleted);
           it < std::begin(reqsCompleted) + nReqCompleted;
           ++it) {
        auto reqCompleted = *it;

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

template <class InputIt, class OutputIt, class Op>
inline void hypercube(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  A2A_ASSERT(nr > 0);

  auto isPower2 = isPow2(static_cast<unsigned>(nr));

  if (!isPower2) {
    return;
  }

  auto trace = TimeTrace{me, "Hypercube"};
  trace.tick(COMMUNICATION);

  auto partner = me ^ 1;

  auto reqs = a2a::sendrecv(
      std::next(begin, partner * blocksize),
      blocksize,
      partner,
      100,
      std::next(out, partner * blocksize),
      blocksize,
      partner,
      100,
      comm);

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  for (int i = 2; i < nr; ++i) {
    partner = me ^ i;

    // Wait for previous round
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    reqs = a2a::sendrecv(
        std::next(begin, partner * blocksize),
        blocksize,
        partner,
        100,
        std::next(out, partner * blocksize),
        blocksize,
        partner,
        100,
        comm);
  }

  // Wait for final round
  A2A_ASSERT_RETURNS(
      MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE), MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);
  // merging
  trace.tock(MERGE);
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  int nr, me;
  MPI_Comm_size(comm, &nr);
  MPI_Comm_rank(comm, &me);

  auto trace = TimeTrace{me, "AlltoAll"};

  trace.tick(COMMUNICATION);

  A2A_ASSERT_RETURNS(
      MPI_Alltoall(
          std::addressof(*begin),
          blocksize,
          mpi_datatype,
          std::addressof(*out),
          blocksize,
          mpi_datatype,
          comm),
      MPI_SUCCESS);

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

        __gnu_parallel::multiway_merge(
            std::begin(seqs),
            std::end(seqs),
            res,
            nels,
            std::less<value_t>{});

#endif

  trace.tock(MERGE);
}
}  // namespace a2a
#endif
