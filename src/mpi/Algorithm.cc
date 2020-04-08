#include <fmpi/mpi/Algorithm.hpp>

namespace mpi {

int isend(
    void const*  buf,
    std::size_t  count,
    MPI_Datatype type,
    Rank         target,
    int          tag,
    MPI_Comm     comm,
    MPI_Request* req) {
  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Isend(
      buf, static_cast<int>(count), type, target, tag, comm, req);
}

int irecv(
    void*        buf,
    std::size_t  count,
    MPI_Datatype type,
    Rank         source,
    int          tag,
    MPI_Comm     comm,
    MPI_Request* req) {
  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Irecv(
      buf, static_cast<int>(count), type, source, tag, comm, req);
}
}  // namespace mpi
