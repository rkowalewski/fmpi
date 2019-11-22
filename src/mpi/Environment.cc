#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>

#include <iosfwd>

namespace mpi {

Context::Context(MPI_Comm comm)
  : m_comm(comm)
{
  int sz;

  int rank;
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &sz), MPI_SUCCESS);
  m_size = sz;

  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &rank), MPI_SUCCESS);
  m_rank = static_cast<Rank>(rank);
}

auto Context::rank() const noexcept -> Rank
{
  return m_rank;
}

auto Context::size() const noexcept -> Context::size_type
{
  return m_size;
}

auto Context::mpiComm() const noexcept -> MPI_Comm
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

auto Rank::operator++() -> Rank&
{
  ++m_rank;
  return *this;
}

auto Rank::operator++(int) const -> const Rank
{
  auto tmp = *this;
  return ++tmp;
}

auto operator==(Rank const& lhs, Rank const& rhs) noexcept -> bool
{
  return static_cast<mpi_rank>(lhs) == static_cast<mpi_rank>(rhs);
}

auto operator!=(Rank const& lhs, Rank const& rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

auto operator<(Rank const& lhs, Rank const& rhs) noexcept -> bool
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return l < r;
}

auto operator>(Rank const& lhs, Rank const& rhs) noexcept -> bool
{
  return !(lhs < rhs) && !(lhs == rhs);
}

auto operator+(Rank const& lhs, Rank const& rhs) noexcept -> Rank
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l + r};
}

auto operator-(Rank const& lhs, Rank const& rhs) noexcept -> Rank
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l - r};
}

auto operator^(Rank const& lhs, Rank const& rhs) -> Rank RTLX_NOEXCEPT
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);

  RTLX_ASSERT(l >= 0);
  RTLX_ASSERT(r >= 0);

  return static_cast<Rank>(
      static_cast<unsigned>(l) ^ static_cast<unsigned>(r));
}

auto operator%(Rank const& lhs, Rank const& rhs) noexcept -> Rank
{
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l % r};
}

auto splitSharedComm(Context const& baseComm) -> Context
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

auto operator<<(std::ostream& os, Rank const& p) -> std::ostream&
{
  os << static_cast<mpi_rank>(p);
  return os;
}

}  // namespace mpi
