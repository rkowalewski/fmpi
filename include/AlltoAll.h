#ifndef BRUCK_H__INCLUDED
#define BRUCK_H__INCLUDED

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>

// Other AllToAll Algorithms
#include <Bruck.h>
#include <Factor.h>

namespace alltoall {

template <class InputIt, class OutputIt>
inline void flatHandshake(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  constexpr auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::value;

  for (int i = 1; i < nr; ++i) {
    auto pair = std::make_pair(mod(me + i, nr), mod(me - i, nr));
    auto res  = MPI_Sendrecv(
        std::addressof(*(begin + pair.first * blocksize)),
        blocksize,
        mpi_datatype,
        pair.first,
        100,
        std::addressof(*(out + pair.second * blocksize)),
        blocksize,
        mpi_datatype,
        pair.second,
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

template <class InputIt, class OutputIt>
inline void hypercube(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  ASSERT(nr > 0);

  auto isPower2 = (nr & (nr - 1)) == 0;

  if (!isPower2) {
    return;
  }

  constexpr auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::value;

  for (int i = 1; i < nr; ++i) {
    auto partner = me ^ i;
    auto res     = MPI_Sendrecv(
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
    ASSERT(res == MPI_SUCCESS);
  }

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);
}

template <class InputIt, class OutputIt>
inline void MpiAlltoAll(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  constexpr auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::value;

  auto res = MPI_Alltoall(
      std::addressof(*begin),
      blocksize,
      mpi_datatype,
      std::addressof(*out),
      blocksize,
      mpi_datatype,
      MPI_COMM_WORLD);

  ASSERT(res == MPI_SUCCESS);
}
}  // namespace alltoall
#endif
