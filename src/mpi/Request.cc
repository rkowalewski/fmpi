#include <fmpi/Memory.h>
#include <fmpi/mpi/Request.h>

namespace mpi {

bool waitall(MPI_Request* begin, MPI_Request* end)
{
  auto n = std::distance(begin, end);
  return MPI_Waitall(n, begin, MPI_STATUSES_IGNORE) == MPI_SUCCESS;
}

int* waitsome(MPI_Request* begin, MPI_Request* end, int* indices)
{
  int completed;

  auto const n = std::distance(begin, end);

  if (!n) return indices;

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(n, begin, &completed, indices, MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  return indices + completed;
}

int* testsome(MPI_Request* begin, MPI_Request* end, int* indices)
{
  int completed;

  auto const n = std::distance(begin, end);

  if (!n) return indices;

  RTLX_ASSERT_RETURNS(
      MPI_Testsome(n, begin, &completed, indices, MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  return indices + completed;
}

}  // namespace mpi
