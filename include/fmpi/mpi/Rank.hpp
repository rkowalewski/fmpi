#ifndef FMPI_MPI_RANK_HPP
#define FMPI_MPI_RANK_HPP

#include <mpi.h>

#include <iosfwd>
#include <type_traits>

#include <rtlx/Assert.hpp>

namespace mpi {

using mpi_rank = int32_t;

struct Rank {
 public:
  Rank() = default;
  explicit Rank(mpi_rank rank) noexcept;
           operator mpi_rank() const noexcept;  // NOLINT
  explicit operator bool() const noexcept;

  auto operator++() -> Rank&;
  auto operator++(int) const -> const Rank;

 private:
  mpi_rank m_rank{MPI_PROC_NULL};
};

auto operator+(Rank const& lhs, Rank const& rhs) noexcept -> Rank;
auto operator-(Rank const& lhs, Rank const& rhs) noexcept -> Rank;
auto operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT -> Rank;
auto operator%(Rank const& lhs, Rank const& rhs) noexcept -> Rank;

auto operator==(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator!=(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator>(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator<(Rank const& lhs, Rank const& rhs) noexcept -> bool;

auto operator<<(std::ostream& os, Rank const& p) -> std::ostream&;
}  // namespace mpi

namespace std {
/// Specialization of std traits for mpi::Rank
template <>
struct is_signed<mpi::Rank> : std::true_type {};
template <>
struct is_integral<mpi::Rank> : std::true_type {};
}  // namespace std
#endif
