#ifndef MPI__COLLECTIVE_H
#define MPI__COLLECTIVE_H

#include <fusion/mpi/Types.h>
#include <fusion/mpi/Environment.h>

namespace mpi {

template <class T>
inline auto mpiAllReduceMinMax(MpiCommCtx const& ctx, T value)
{
  auto mpi_type = mpi::type_mapper<T>::type();

  T min, max;

  A2A_ASSERT_RETURNS(
      MPI_Allreduce(&value, &min, 1, mpi_type, MPI_MIN, ctx.mpiComm()),
      MPI_SUCCESS);
  A2A_ASSERT_RETURNS(
      MPI_Allreduce(&value, &max, 1, mpi_type, MPI_MAX, ctx.mpiComm()),
      MPI_SUCCESS);

  return std::make_pair(min, max);
}

}

#endif