#include <fmpi/mpi/Algorithm.h>
#include <iterator>

#include <rtlx/Assert.h>

inline MPI_Request* waitsome(MPI_Request* begin, MPI_Request* end)
{
  auto pending = std::distance(begin, end);

  std::vector<int> indices(nreqs, MPI_UNDEFINED);

  int completed;

  RTLX_ASSERT_RETURNS(
      MPI_Waitsome(
          pending, begin, completed, &(indices[0]), MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

  if (completed == MPI_UNDEFINED) {
    return begin;
  }

}
