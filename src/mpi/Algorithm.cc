#include <fmpi/mpi/Algorithm.h>

namespace mpi {

auto isend_type(
    void const*    buf,
    std::size_t    count,
    MPI_Datatype   type,
    Rank           target,
    int            tag,
    Context const& ctx,
    MPI_Request*   req) -> bool
{
  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Isend(
             buf,
             static_cast<int>(count),
             type,
             target,
             tag,
             ctx.mpiComm(),
             req) == MPI_SUCCESS;
}

auto irecv_type(
    void*          buf,
    std::size_t    count,
    MPI_Datatype   type,
    Rank           source,
    int            tag,
    Context const& ctx,
    MPI_Request*   req) -> bool
{
  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Irecv(
             buf,
             static_cast<int>(count),
             type,
             source,
             tag,
             ctx.mpiComm(),
             req) == MPI_SUCCESS;
}
}  // namespace mpi
