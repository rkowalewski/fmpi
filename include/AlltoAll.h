#ifndef BRUCK_H__INCLUDED
#define BRUCK_H__INCLUDED

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <parallel/algorithm>

// Other AllToAll Algorithms
#include <Bruck.h>
#include <Debug.h>
#include <Factor.h>
#include <Trace.h>

static constexpr char MERGE[]         = "merge";
static constexpr char COMMUNICATION[] = "communication";

namespace a2a {

template <class InputIt, class OutputIt, class Op>
inline void flatHandshake(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  auto trace = TimeTrace{me, "FlatHandshake"};

  trace.tick(COMMUNICATION);

  for (int i = 1; i < nr; ++i) {
    auto pair = std::make_pair(mod(me + i, nr), mod(me - i, nr));
    auto dst  = pair.first;
    auto src  = pair.second;
    A2A_ASSERT_RETURNS(
        MPI_Sendrecv(
            std::addressof(*(begin + pair.first * blocksize)),
            blocksize,
            mpi_datatype,
            dst,
            100,
            std::addressof(*(out + pair.second * blocksize)),
            blocksize,
            mpi_datatype,
            src,
            100,
            comm,
            MPI_STATUS_IGNORE),
        MPI_SUCCESS);
  }

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

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

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  auto trace = TimeTrace{me, "Hypercube"};
  trace.tick(COMMUNICATION);

  std::array<MPI_Request, 2> reqs = {MPI_REQUEST_NULL};

  auto partner = me ^ 1;

  // Overlapping first round...
  A2A_ASSERT_RETURNS(
      MPI_Irecv(
          std::addressof(*(out + partner * blocksize)),
          blocksize,
          mpi_datatype,
          partner,
          100,
          comm,
          &(reqs[1])),
      MPI_SUCCESS);

  A2A_ASSERT_RETURNS(
      MPI_Isend(
          std::addressof(*(begin + partner * blocksize)),
          blocksize,
          mpi_datatype,
          partner,
          100,
          comm,
          &(reqs[0])),
      MPI_SUCCESS);

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  for (int i = 2; i < nr; ++i) {
    partner = me ^ i;

    // Wait for previous round
    A2A_ASSERT_RETURNS(
        MPI_Waitall(2, &(reqs[0]), MPI_STATUSES_IGNORE), MPI_SUCCESS);

#if 0
    auto res = MPI_Sendrecv(
        std::addressof(*(begin + partner * blocksize)),
        blocksize,
        mpi_datatype,
        partner,
        100,
        std::addressof(*(out + partner * blocksize)),
        blocksize,
        mpi_datatype,
        partner,
        100,
        comm,
        MPI_STATUS_IGNORE);
#else

    A2A_ASSERT_RETURNS(
        MPI_Irecv(
            std::addressof(*(out + partner * blocksize)),
            blocksize,
            mpi_datatype,
            partner,
            100,
            comm,
            &(reqs[1])),
        MPI_SUCCESS);

    A2A_ASSERT_RETURNS(
        MPI_Isend(
            std::addressof(*(begin + partner * blocksize)),
            blocksize,
            mpi_datatype,
            partner,
            100,
            comm,
            &(reqs[0])),
        MPI_SUCCESS);
#endif
  }

  // Wait for final round
  A2A_ASSERT_RETURNS(
      MPI_Waitall(2, &(reqs[0]), MPI_STATUSES_IGNORE), MPI_SUCCESS);

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
