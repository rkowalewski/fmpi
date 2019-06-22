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

template <class InputIt, class OutputIt, class Op, size_t NReqs>
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
    s = std::move(os.str());
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

    for (auto i = 0; i < ss; ++i) {
      // Overlapping first round...
      auto recvfrom = mod(me - i + ii, nr);

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

template <class InputIt, class OutputIt, class Op>
inline void hypercube(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  A2A_ASSERT(nr > 0);

  auto isPower2 = (nr & (nr - 1)) == 0;

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

#endif

  trace.tock(MERGE);
}
}  // namespace a2a
#endif
