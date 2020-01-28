#include <fmpi/mpi/Environment.hpp>

namespace mpi {

Rank::Rank(int32_t rank) noexcept
  : m_rank(rank) {
}

Rank::operator int32_t() const noexcept {
  return m_rank;
}

Rank::operator bool() const noexcept {
  return m_rank != MPI_PROC_NULL && m_rank >= 0;
}

auto Rank::operator++() -> Rank& {
  ++m_rank;
  return *this;
}

auto Rank::operator++(int) const -> const Rank {
  auto tmp = *this;
  return ++tmp;
}

auto operator==(Rank const& lhs, Rank const& rhs) noexcept -> bool {
  return static_cast<mpi_rank>(lhs) == static_cast<mpi_rank>(rhs);
}

auto operator!=(Rank const& lhs, Rank const& rhs) noexcept -> bool {
  return !(lhs == rhs);
}

auto operator<(Rank const& lhs, Rank const& rhs) noexcept -> bool {
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return l < r;
}

auto operator>(Rank const& lhs, Rank const& rhs) noexcept -> bool {
  return !(lhs < rhs) && !(lhs == rhs);
}

auto operator+(Rank const& lhs, Rank const& rhs) noexcept -> Rank {
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l + r};
}

auto operator-(Rank const& lhs, Rank const& rhs) noexcept -> Rank {
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l - r};
}

auto operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT -> Rank {
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);

  RTLX_ASSERT(l >= 0);
  RTLX_ASSERT(r >= 0);

  return static_cast<Rank>(
      static_cast<unsigned>(l) ^ static_cast<unsigned>(r));
}

auto operator%(Rank const& lhs, Rank const& rhs) noexcept -> Rank {
  auto l = static_cast<mpi_rank>(lhs);
  auto r = static_cast<mpi_rank>(rhs);
  return Rank{l % r};
}

std::ostream& operator<<(std::ostream& os, Rank const& p) {
  os << static_cast<mpi_rank>(p);
  return os;
}

}  // namespace mpi
