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

namespace alltoall {

template <class InputIt, class OutputIt, class Op>
inline void flatHandshake(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::type();

  for (int i = 1; i < nr; ++i) {
    auto pair = std::make_pair(mod(me + i, nr), mod(me - i, nr));
    auto dst  = pair.first;
    auto src  = pair.second;
    auto res  = MPI_Sendrecv(
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
        MPI_STATUS_IGNORE);
    ASSERT(res == MPI_SUCCESS);
  }

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);
}

template <class InputIt, class OutputIt, class Op>
inline void hypercube(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  ASSERT(nr > 0);

  auto isPower2 = (nr & (nr - 1)) == 0;

  if (!isPower2) {
    return;
  }

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  auto recv_buf =
      std::unique_ptr<value_type[]>(new value_type[blocksize * 2]);
  auto merge_buf =
      std::unique_ptr<value_type[]>(new value_type[blocksize * nr]);

  auto* it_out  = &(*out);
  auto* it_mbuf = merge_buf.get();

  auto reqs = std::array<MPI_Request, 2>{MPI_REQUEST_NULL};

  auto partner = me ^ 1;
  auto half    = 0;

  MPI_Irecv(
      recv_buf.get() + half * blocksize,
      blocksize,
      mpi_datatype,
      partner,
      100,
      comm,
      &reqs[0]);

  MPI_Send(
      &(*(begin + partner * blocksize)),
      blocksize,
      mpi_datatype,
      partner,
      100,
      comm);

  std::copy(
      begin + me * blocksize, begin + me * blocksize + blocksize, it_out);

  auto recvcount = blocksize;

  for (int i = 2; i < nr; ++i) {
#if 0
    auto res = MPI_Sendrecv(
        std::addressof(*(begin + partner * blocksize)),
        blocksize,
        mpi_datatype,
        partner,
        100,
        recv_buf.get(),
        blocksize,
        mpi_datatype,
        partner,
        100,
        comm,
        MPI_STATUS_IGNORE);
#endif

    auto res = MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    ASSERT(res == MPI_SUCCESS);

    // next round
    partner       = me ^ i;
    auto prevHalf = half;
    half          = 1 - prevHalf;

    MPI_Irecv(
        recv_buf.get() + half * blocksize,
        blocksize,
        mpi_datatype,
        partner,
        100,
        comm,
        &reqs[0]);

    MPI_Send(
        &(*(begin + partner * blocksize)),
        blocksize,
        mpi_datatype,
        partner,
        100,
        comm);

    op(it_out,
       it_out + recvcount,
       recv_buf.get() + prevHalf * blocksize,
       recv_buf.get() + (prevHalf + 1) * blocksize,
       it_mbuf);

    recvcount += blocksize;

    std::swap(it_out, it_mbuf);
  }

  auto res = MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
  ASSERT(res == MPI_SUCCESS);

  op(it_out,
     it_out + recvcount,
     recv_buf.get() + half * blocksize,
     recv_buf.get() + (half + 1) * blocksize,
     it_mbuf);

  recvcount += blocksize;

  if (&(*it_mbuf) != &(*out)) {
    std::copy(it_mbuf, it_mbuf + recvcount, out);
  }
  ASSERT(recvcount == blocksize * nr);
  ASSERT(std::is_sorted(it_out, it_out + nr * blocksize));
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto mpi_datatype = mpi::mpi_datatype<value_type>::type();

  int nr;
  MPI_Comm_size(comm, &nr);

  auto res = MPI_Alltoall(
      std::addressof(*begin),
      blocksize,
      mpi_datatype,
      std::addressof(*out),
      blocksize,
      mpi_datatype,
      comm);

  ASSERT(res == MPI_SUCCESS);

#if 1
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
#else
  std::sort(out, out + blocksize * nr);
#endif
}
}  // namespace alltoall
#endif
