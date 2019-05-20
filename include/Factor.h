#ifndef FACTOR_H
#define FACTOR_H

#include <mpi.h>

#include <memory>

#include <Debug.h>
#include <Math.h>
#include <Types.h>

namespace alltoall {

template <class InputIt, class OutputIt>
inline void factorParty(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  constexpr auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::value;

  ASSERT(nr % 2 == 0);

  std::unique_ptr<int[]> partner{new int[nr]};

  // We have 2n ranks
  auto n = nr / 2;

  auto factorPair = [n](int me, int r) {
    auto k_bottom = 2 * n - 1;
    return std::make_pair(
        mod(r + me, k_bottom) + 1, mod(r - me, k_bottom) + 1);
  };

  // Rounds
  for (int r = 1; r < nr; ++r) {
    partner[0] = r;
    partner[r] = 0;

    // generate remaining pairs
    for (int p = 1; p < n; ++p) {
      auto pair            = factorPair(p, r - 1);
      partner[pair.first]  = pair.second;
      partner[pair.second] = pair.first;
    }

    auto res = MPI_Sendrecv(
        std::addressof(*(begin + partner[me] * blocksize)),
        blocksize,
        mpi_datatype,
        partner[me],
        100,
        std::addressof(*(out + partner[me] * blocksize)),
        blocksize,
        mpi_datatype,
        partner[me],
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
inline void flatFactor(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  constexpr auto mpi_datatype = mpi::mpi_datatype<
      typename std::iterator_traits<InputIt>::value_type>::value;

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
}  // namespace alltoall

#endif
