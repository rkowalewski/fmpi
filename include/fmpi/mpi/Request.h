#ifndef MPI__REQUEST_H
#define MPI__REQUEST_H

#include <mpi.h>

#include <array>
#include <iterator>
#include <vector>

#include <rtlx/Assert.h>

namespace mpi {

template <std::size_t N>
inline std::vector<int> waitsome(std::array<MPI_Request, N> pending)
{
  std::array<int, N> indices{};

  int completed;

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(
          N, &(pending[0]), &completed, &(indices[0]), MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  return std::vector<int>(
      std::begin(indices), std::next(std::begin(indices), completed));
}
}  // namespace mpi

#endif
