#ifndef FMPI_MPI_RANK_HPP
#define FMPI_MPI_RANK_HPP

#include <mpi.h>

#include <iosfwd>
#include <type_traits>

#include <fmpi/detail/Assert.hpp>

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
  constexpr const Rank operator++(int) noexcept;
  // Prefix Decrement
  constexpr Rank operator--() noexcept;
  // Postfix Decrement
  constexpr const Rank operator--(int) noexcept;

  [[nodiscard]] constexpr int mpiRank() const noexcept;

  static constexpr Rank null() {
    return mpi::Rank{MPI_PROC_NULL};
  }

 private:
  int m_rank{MPI_PROC_NULL};
};

constexpr Rank operator+(Rank const& lhs, Rank const& rhs) noexcept;
constexpr Rank operator-(Rank const& lhs, Rank const& rhs) noexcept;
constexpr Rank operator^(Rank const& lhs, Rank const& rhs) FMPI_NOEXCEPT;
constexpr Rank operator%(Rank const& lhs, Rank const& rhs) noexcept;

constexpr bool operator==(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator!=(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator>(Rank const& lhs, Rank const& rhs) noexcept;
constexpr bool operator<(Rank const& lhs, Rank const& rhs) noexcept;

auto operator<<(std::ostream& os, Rank const& p) -> std::ostream&;
}  // namespace mpi

//////////////////////////////////////////////////////
// Implementation ////////////////////////////////////
//////////////////////////////////////////////////////

namespace mpi {

constexpr Rank::Rank(int32_t rank) noexcept
  : m_rank(rank) {
}

constexpr Rank::operator int32_t() const noexcept {
  return mpiRank();
}

constexpr int Rank::mpiRank() const noexcept {
  return m_rank;
}

constexpr Rank::operator bool() const noexcept {
  return m_rank != MPI_PROC_NULL && m_rank >= 0;
}

constexpr Rank Rank::operator++() noexcept {
  ++m_rank;
  return *this;
}

constexpr const Rank Rank::operator++(int) noexcept {
  auto tmp = *this;
  ++m_rank;
  return tmp;
}

constexpr Rank Rank::operator--() noexcept {
  --m_rank;
  return *this;
}

constexpr const Rank Rank::operator--(int) noexcept {
  auto tmp = *this;
  --m_rank;
  return tmp;
}

constexpr bool operator==(Rank const& lhs, Rank const& rhs) noexcept {
  return lhs.mpiRank() == rhs.mpiRank();
}

constexpr bool operator!=(Rank const& lhs, Rank const& rhs) noexcept {
  return !(lhs == rhs);
}

constexpr bool operator<(Rank const& lhs, Rank const& rhs) noexcept {
  return lhs.mpiRank() < rhs.mpiRank();
}

constexpr bool operator>(Rank const& lhs, Rank const& rhs) noexcept {
  return lhs.mpiRank() > rhs.mpiRank();
}

constexpr Rank operator+(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpiRank() + rhs.mpiRank()};
}

constexpr Rank operator-(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpiRank() - rhs.mpiRank()};
}

constexpr Rank operator^(Rank const& lhs, Rank const& rhs) FMPI_NOEXCEPT {
  FMPI_ASSERT(lhs.mpiRank() >= 0);
  FMPI_ASSERT(rhs.mpiRank() >= 0);

  auto xor_res = static_cast<unsigned>(lhs.mpiRank()) ^
                 static_cast<unsigned>(rhs.mpiRank());

  return Rank{static_cast<int>(xor_res)};
}

constexpr Rank operator%(Rank const& lhs, Rank const& rhs) noexcept {
  return Rank{lhs.mpiRank() % rhs.mpiRank()};
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
