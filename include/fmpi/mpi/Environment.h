#ifndef FMPI_MPI_ENVIRONMENT_H
#define FMPI_MPI_ENVIRONMENT_H

#include <cstdint>
#include <type_traits>

#include <mpi.h>

#include <fmpi/mpi/TypeMapper.h>

#include <rtlx/Assert.h>

#include <iosfwd>

namespace mpi {

using mpi_rank = int32_t;

struct Rank {
 public:
  Rank() = default;
  explicit Rank(mpi_rank rank) noexcept;
           operator mpi_rank() const noexcept;  // NOLINT
  explicit operator bool() const noexcept;

  auto      operator++() -> Rank&;
  auto operator++(int) const -> const Rank;

 private:
  mpi_rank m_rank{MPI_PROC_NULL};
};

auto operator+(Rank const& lhs, Rank const& rhs) noexcept -> Rank;
auto operator-(Rank const& lhs, Rank const& rhs) noexcept -> Rank;
auto operator^(Rank const& lhs, Rank const& rhs) -> Rank RTLX_NOEXCEPT;
auto operator%(Rank const& lhs, Rank const& rhs) noexcept -> Rank;

auto operator==(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator!=(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator>(Rank const& lhs, Rank const& rhs) noexcept -> bool;
auto operator<(Rank const& lhs, Rank const& rhs) noexcept -> bool;

auto operator<<(std::ostream& os, Rank const& p) -> std::ostream&;

class Context {
 public:
  using size_type = std::uint32_t;

 public:
  explicit Context(MPI_Comm comm);

  auto rank() const noexcept -> Rank;

  auto size() const noexcept -> size_type;

  auto mpiComm() const noexcept -> MPI_Comm;

 private:
  size_type m_size{};
  Rank      m_rank{};
  MPI_Comm  m_comm{MPI_COMM_NULL};
};

auto splitSharedComm(Context const& baseComm) -> Context;

}  // namespace mpi

namespace std {
template <>
struct is_signed<mpi::Rank> : std::true_type {
};
template <>
struct is_integral<mpi::Rank> : std::true_type {
};
}  // namespace std

#endif
