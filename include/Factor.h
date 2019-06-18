#ifndef FACTOR_H
#define FACTOR_H

#include <mpi.h>

#include <cassert>
#include <iterator>
#include <memory>

#include <Constants.h>
#include <Debug.h>
#include <Math.h>
#include <Mpi.h>
#include <NumericRange.h>
#include <Trace.h>
#include <Types.h>

namespace a2a {

namespace detail {

template <class InputIt, class OutputIt, class Op>
inline void oneFactor_odd(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert(nr % 2);

  auto steps = a2a::range(1, nr + 1);

  auto trace = TimeTrace{me, "OneFactor"};
  trace.tick(COMMUNICATION);

  for (auto const r : steps) {
    auto partner = mod(r - me, nr);

    if (partner == me) {
      std::copy(
          begin + me * blocksize,
          begin + me * blocksize + blocksize,
          out + me * blocksize);
    }
    else {
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

      A2A_ASSERT_RETURNS(
          MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
          MPI_SUCCESS);
    }
  }
  trace.tock(COMMUNICATION);
}

template <class InputIt, class OutputIt, class Op>
inline void oneFactor_even(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&&)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert((nr % 2) == 0);

  auto partner = [nr, me](auto step) {
    auto idle = mod(nr * step / 2, nr - 1);

    if (me == nr - 1) {
      return idle;
    }
    else if (me == idle) {
      return nr - 1;
    }
    else {
      return mod(step - me, nr - 1);
    }
  };

  auto trace = TimeTrace{me, "OneFactor"};
  trace.tick(COMMUNICATION);
  auto p = partner(0);

  auto reqs = a2a::sendrecv(
      std::next(begin, p * blocksize),
      blocksize,
      p,
      100,
      std::next(out, p * blocksize),
      blocksize,
      p,
      100,
      comm);

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  for (int r = 1; r < nr - 1; ++r) {
    p = partner(r);

    // Wait for previous round
    A2A_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    reqs = a2a::sendrecv(
        std::next(begin, p * blocksize),
        blocksize,
        p,
        100,
        std::next(out, p * blocksize),
        blocksize,
        p,
        100,
        comm);
  }

  // Final Round
  A2A_ASSERT_RETURNS(
      MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE), MPI_SUCCESS);
  trace.tock(COMMUNICATION);
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

}  // namespace a2a

#endif
