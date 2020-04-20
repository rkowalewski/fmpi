#ifndef FMPI_MPI_RANK_HPP
#define FMPI_MPI_RANK_HPP

#include <mpi.h>

#include <iosfwd>
#include <type_traits>

#include <rtlx/Assert.hpp>

namespace mpi {

class Rank {
 public:
  constexpr Rank() = default;
  constexpr explicit Rank(int rank) noexcept;
  constexpr          operator int() const noexcept;  // NOLINT
  constexpr explicit operator bool() const noexcept;

  // Prefix Increment
  constexpr Rank operator++() noexcept;
  // Postfix Increment
  constexpr Rank operator++(int) const noexcept;
  // Prefix Decrement
  constexpr Rank operator--() noexcept;
  // Postfix Decrement
  constexpr Rank operator--(int) const noexcept;

  constexpr int mpi_rank() const noexcept;

 private:
  int m_rank{MPI_PROC_NULL};
};

constexpr Rank operator+(Rank const& lhs, Rank const& rhs) noexcept;
constexpr Rank operator-(Rank const& lhs, Rank const& rhs) noexcept;
constexpr Rank operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT;
constexpr Rank operator%(Rank const& lhs, Rank const& rhs) noexcept;

constexpr bool operator==(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator!=(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator>(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator<(Rank const& lhs, Rank const& rhs) noexcept;

auto operator<<(std::ostream& os, Rank const& p) -> std::ostream&;

constexpr Rank::Rank(int32_t rank) noexcept
  : m_rank(rank) {
}

constexpr Rank::operator int32_t() const noexcept {
  return mpi_rank();
}

constexpr int Rank::mpi_rank() const noexcept {
  return m_rank;
}

constexpr Rank::operator bool() const noexcept {
  return m_rank != MPI_PROC_NULL && m_rank >= 0;
}

constexpr Rank Rank::operator++() noexcept {
  ++m_rank;
  return *this;
}

constexpr Rank Rank::operator++(int) const noexcept {
  auto tmp = *this;
  return ++tmp;
}

constexpr bool operator==(Rank const& lhs, Rank const& rhs) noexcept {
  return lhs.mpi_rank() == rhs.mpi_rank();
}

constexpr bool operator!=(Rank const& lhs, Rank const& rhs) noexcept {
  return !(lhs == rhs);
}

constexpr bool operator<(Rank const& lhs, Rank const& rhs) noexcept {
  return lhs.mpi_rank() < rhs.mpi_rank();
}

constexpr bool operator>(Rank const& lhs, Rank const& rhs) noexcept {
  return !(lhs < rhs) && !(lhs == rhs);
}

constexpr Rank operator+(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpi_rank() + rhs.mpi_rank()};
}

constexpr Rank operator-(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpi_rank() - rhs.mpi_rank()};
}

constexpr Rank operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT {
  RTLX_ASSERT(lhs.mpi_rank() >= 0);
  RTLX_ASSERT(rhs.mpi_rank() >= 0);

  auto xor_res = static_cast<unsigned>(lhs.mpi_rank()) ^
                 static_cast<unsigned>(rhs.mpi_rank());

  return Rank{static_cast<int>(xor_res)};
}

constexpr Rank operator%(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpi_rank() % rhs.mpi_rank()};
}

}  // namespace mpi

namespace std {
/// Specialization of std traits for mpi::Rank
template <>
struct is_signed<mpi::Rank> : std::true_type {};
template <>
struct is_integral<mpi::Rank> : std::true_type {};
}  // namespace std
#endif
