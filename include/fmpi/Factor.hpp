#ifndef FACTOR_H
#define FACTOR_H

#include <mpi.h>

#include <cassert>
#include <iterator>
#include <memory>
#include <vector>

#include <Constants.h>
#include <Debug.h>
#include <Math.h>
#include <Mpi.h>
#include <NumericRange.h>
#include <Trace.h>
#include <Types.h>

namespace fmpi {

namespace detail {

template <class InputIt, class OutputIt, class Op, std::size_t NReqs = 1>
inline void oneFactor_odd(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& /*op*/)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert(nr % 2);

  auto steps = fmpi::range(0, nr);

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
      auto reqs = fmpi::sendrecv(
          std::next(begin, partner * blocksize),
          blocksize,
          partner,
          100,
          std::next(out, partner * blocksize),
          blocksize,
          partner,
          100,
          comm);

      RTLX_ASSERT_RETURNS(
          MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
          MPI_SUCCESS);
    }
  }
  trace.tock(COMMUNICATION);
}

template <class InputIt, class OutputIt, class Op, std::size_t NReqs = 1>
inline void oneFactor_even(
    InputIt  begin,
    OutputIt out,
    int      blocksize,
    MPI_Comm comm,
    Op&& /*unused*/)
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
    if (me == idle) {
      return nr - 1;
    }

    return mod(step - me, nr - 1);
  };

  auto trace = TimeTrace{me, "OneFactor"};
  trace.tick(COMMUNICATION);

  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      out + me * blocksize);

  for (int r = 0; r < nr - 1; ++r) {
    auto p = partner(r);

    auto reqs = fmpi::sendrecv(
        std::next(begin, p * blocksize),
        blocksize,
        p,
        100,
        std::next(out, p * blocksize),
        blocksize,
        p,
        100,
        comm);

    // Wait for previous round
    RTLX_ASSERT_RETURNS(
        MPI_Waitall(reqs.size(), &(reqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);
  }

  trace.tock(COMMUNICATION);
}
}  // namespace detail

template <class InputIt, class OutputIt, class Op, size_t NReqs = 1>
inline void oneFactor(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
{
  int nr;
  MPI_Comm_size(comm, &nr);

  using value_type  = typename std::iterator_traits<InputIt>::value_type;
  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  if (nr % 2) {
    detail::oneFactor_odd(begin, &rbuf[0], blocksize, comm, std::forward<Op>(op));
  }
  else {
    detail::oneFactor_even(begin, &rbuf[0], blocksize, comm, std::forward<Op>(op));
  }


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
}

}  // namespace fmpi

#endif
