#include <fmpi/mpi/Request.h>
#include <rtlx/Assert.h>
#include <iterator>

inline std::vector<int> waitsome(MPI_Request* begin, MPI_Request* end)
{
  auto pending = std::distance(begin, end);

  std::vector<int> indices(pending);

  int completed;

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(
          pending, begin, completed, &(indices[0]), MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  indices.resize(completed);

  return indices;
}
