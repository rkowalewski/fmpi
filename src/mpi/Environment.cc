#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>

#include <iosfwd>

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

Rank::operator bool() const noexcept
{
  return m_rank != MPI_PROC_NULL && m_rank >= 0;
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

bool operator==(Rank const& lhs, Rank const& rhs) noexcept
{
  return static_cast<mpi_rank>(lhs) == static_cast<mpi_rank>(rhs);
}

bool operator!=(Rank const& lhs, Rank const& rhs) noexcept
{
  return !(lhs == rhs);
}

bool operator<(Rank const& lhs, Rank const& rhs) noexcept
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return l < r;
}

bool operator>(Rank const& lhs, Rank const& rhs) noexcept
{
  return !(lhs < rhs) && !(lhs == rhs);
}

Rank operator+(Rank const& lhs, Rank const& rhs) noexcept
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l + r};
}

Rank operator-(Rank const& lhs, Rank const& rhs) noexcept
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l - r};
}

Rank operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);

  RTLX_ASSERT(l >= 0);
  RTLX_ASSERT(r >= 0);

  return static_cast<Rank>(
      static_cast<unsigned>(l) ^ static_cast<unsigned>(r));
}

Rank operator%(Rank const& lhs, Rank const& rhs) noexcept
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l % r};
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

std::ostream& operator<<(std::ostream& os, Rank const& p)
{
  os << static_cast<mpi_rank>(p);
  return os;
}

}  // namespace mpi
