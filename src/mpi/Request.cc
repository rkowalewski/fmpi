#include <fmpi/mpi/Request.h>

namespace mpi {

bool waitall(MPI_Request* begin, MPI_Request* end)
{
  auto n = std::distance(begin, end);
  return MPI_Waitall(n, begin, MPI_STATUSES_IGNORE) == MPI_SUCCESS;
}

}  // namespace mpi
