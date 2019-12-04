#include <fmpi/Memory.hpp>
#include <fmpi/mpi/Request.hpp>

namespace mpi {

auto waitall(MPI_Request* begin, MPI_Request* end) -> bool
{
  auto n = std::distance(begin, end);
  return MPI_Waitall(n, begin, MPI_STATUSES_IGNORE) == MPI_SUCCESS;
}

auto waitsome(MPI_Request* begin, MPI_Request* end, int* indices) -> int*
{
  int completed;

  auto const n = std::distance(begin, end);

  if (n == 0) {
    return indices;
  }

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(n, begin, &completed, indices, MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  return std::next(indices, completed);
}

auto testsome(MPI_Request* begin, MPI_Request* end, int* indices) -> int*
{
  int completed;

  auto const n = std::distance(begin, end);

  if (n == 0) {
    return indices;
  }

  RTLX_ASSERT_RETURNS(
      MPI_Testsome(n, begin, &completed, indices, MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  return std::next(indices, completed);
}

}  // namespace mpi
