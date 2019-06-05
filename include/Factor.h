#ifndef FACTOR_H
#define FACTOR_H

#include <mpi.h>

#include <cassert>
#include <memory>

#include <Debug.h>
#include <Math.h>
#include <Types.h>

namespace alltoall {

namespace detail {

template <class InputIt, class OutputIt, class Op>
inline void oneFactor_odd(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert(nr % 2);

  auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::type();

  for (int i = 1; i <= nr; ++i) {
    auto partner = mod(i - me, nr);

    if (partner == me) {
      std::copy(
          begin + me * blocksize,
          begin + me * blocksize + blocksize,
          out + me * blocksize);
    }
    else {
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
      ASSERT(res == MPI_SUCCESS);
    }
  }
}

template <class InputIt, class OutputIt, class Op>
inline void oneFactor_even(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&&)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert((nr % 2) == 0);

  auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::type();

  for (int r = 0; r < nr - 1; ++r) {
    auto idle = mod(nr * r / 2, nr - 1);
    int  partner;

    if (me == nr - 1) {
      partner = idle;
    }
    else if (me == idle) {
      partner = nr - 1;
    }
    else {
      partner = mod(r - me, nr - 1);
    }

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
    ASSERT(res == MPI_SUCCESS);
  }

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);
}
}  // namespace detail

template <class InputIt, class OutputIt, class Op>
inline void oneFactor(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
{
  int nr;
  MPI_Comm_size(comm, &nr);
  if (nr % 2) {
    detail::oneFactor_odd(begin, out, blocksize, comm, std::forward<Op>(op));
  }
  else {
    detail::oneFactor_even(begin, out, blocksize, comm, std::forward<Op>(op));
  }
}

}  // namespace alltoall

#endif
