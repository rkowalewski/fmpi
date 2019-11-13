#ifndef MPI__REQUEST_H
#define MPI__REQUEST_H

#include <mpi.h>

#include <array>
#include <iterator>
#include <vector>

#include <rtlx/Assert.h>

#include <fmpi/Debug.h>

namespace mpi {

template <std::size_t N>
inline std::vector<int> waitsome(std::array<MPI_Request, N>& pending)
{
  std::array<int, N> indices{};

  int completed;

  using vector = std::vector<int>;

  FMPI_DBG(pending);

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(
          N, &(pending[0]), &completed, &(indices[0]), MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  if (completed == MPI_UNDEFINED) {
    return vector(0);
  }

  return vector(
      std::begin(indices), std::next(std::begin(indices), completed));
}

bool waitall(MPI_Request* begin, MPI_Request* end);

}  // namespace mpi

#endif
