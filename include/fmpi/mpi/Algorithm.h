#ifndef MPI__ALGORITHM_H
#define MPI__ALGORITHM_H

#include <fmpi/mpi/Environment.h>
#include <fmpi/mpi/Types.h>

namespace mpi {

using return_code = int;

template <class T>
inline return_code irecv(
    T*                buf,
    std::size_t       count,
    rank_t            source,
    int               tag,
    MpiCommCtx const& ctx,
    MPI_Request&      req)

{
  auto type = mpi::type_mapper<T>::type();

  return MPI_Irecv(buf, count, type, source, tag, ctx.mpiComm(), &req);
}

template <class T>
inline return_code isend(
    T const*                buf,
    std::size_t       count,
    rank_t            target,
    int               tag,
    MpiCommCtx const& ctx,
    MPI_Request&      req)

{
  auto type = mpi::type_mapper<T>::type();

  return MPI_Isend(buf, count, type, target, tag, ctx.mpiComm(), &req);
}

template <class T>
inline auto allreduce_minmax(MpiCommCtx const& ctx, T value)
{
  auto mpi_type = mpi::type_mapper<T>::type();

  T min, max;

  RTLX_ASSERT_RETURNS(
      MPI_Allreduce(&value, &min, 1, mpi_type, MPI_MIN, ctx.mpiComm()),
      MPI_SUCCESS);
  RTLX_ASSERT_RETURNS(
      MPI_Allreduce(&value, &max, 1, mpi_type, MPI_MAX, ctx.mpiComm()),
      MPI_SUCCESS);

  return std::make_pair(min, max);
}

}  // namespace mpi

#endif
