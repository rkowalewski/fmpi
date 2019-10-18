#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>

namespace mpi {

MpiCommCtx::MpiCommCtx(MPI_Comm comm)
  : m_comm(comm)
{
  int sz, rank;
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &sz), MPI_SUCCESS);
  m_size = sz;

  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &rank), MPI_SUCCESS);
  m_rank = static_cast<Rank>(rank);
}

Rank MpiCommCtx::rank() const noexcept
{
  return m_rank;
}

MpiCommCtx::size_type MpiCommCtx::size() const noexcept
{
  return m_size;
}

MPI_Comm MpiCommCtx::mpiComm() const noexcept
{
  return m_comm;
}

Rank::Rank(int32_t rank) noexcept
  : m_rank(rank)
{
}

Rank::operator int32_t() const noexcept
{
  return m_rank;
}

bool operator==(Rank lhs, Rank rhs) noexcept
{
  return static_cast<mpi_rank>(lhs) == static_cast<mpi_rank>(rhs);
}

bool operator!=(Rank lhs, Rank rhs) noexcept
{
  return !(lhs == rhs);
}

MpiCommCtx splitSharedComm(MpiCommCtx const& baseComm)
{
  MPI_Comm sharedComm;
  // split world into shared memory communicator
  RTLX_ASSERT_RETURNS(
      MPI_Comm_split_type(
          baseComm.mpiComm(),
          MPI_COMM_TYPE_SHARED,
          0,
          MPI_INFO_NULL,
          &sharedComm),
      MPI_SUCCESS);

  return MpiCommCtx{sharedComm};
}

}  // namespace mpi
