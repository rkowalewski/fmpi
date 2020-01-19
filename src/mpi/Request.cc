#include <fmpi/mpi/Request.hpp>

namespace mpi {

auto waitall(MPI_Request* begin, MPI_Request* end, MPI_Status* statuses)
    -> bool {
  auto n = std::distance(begin, end);
  return MPI_Waitall(n, begin, statuses) == MPI_SUCCESS;
}

int waitsome(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last) {
  int completed;

  auto const n = std::distance(begin, end);

  auto ret = MPI_Waitsome(n, begin, &completed, indices, statuses);

  last =
      (completed != MPI_UNDEFINED) ? std::next(indices, completed) : indices;

  return ret;
}

int testsome(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last) {
  int completed;

  auto const n = std::distance(begin, end);

  auto ret = MPI_Testsome(n, begin, &completed, indices, statuses);

  last =
      (completed != MPI_UNDEFINED) ? std::next(indices, completed) : indices;

  return ret;
}

}  // namespace mpi
