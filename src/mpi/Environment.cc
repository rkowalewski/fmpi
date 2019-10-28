#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>

namespace mpi {

Context::Context(MPI_Comm comm)
  : m_comm(comm)
{
  int sz, rank;
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &sz), MPI_SUCCESS);
  m_size = sz;

  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &rank), MPI_SUCCESS);
  m_rank = static_cast<Rank>(rank);
}

Rank Context::rank() const noexcept
{
  return m_rank;
}

Context::size_type Context::size() const noexcept
{
  return m_size;
}

MPI_Comm Context::mpiComm() const noexcept
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

Rank& Rank::operator++()
{
  ++m_rank;
  return *this;
}

const Rank Rank::operator++(int) const
{
  auto tmp = *this;
  return ++tmp;
}

bool operator==(Rank lhs, Rank rhs) noexcept
{
  return static_cast<mpi_rank>(lhs) == static_cast<mpi_rank>(rhs);
}

bool operator!=(Rank lhs, Rank rhs) noexcept
{
  return !(lhs == rhs);
}

Context splitSharedComm(Context const& baseComm)
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

  return Context{sharedComm};
}

}  // namespace mpi
