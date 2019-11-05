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

  Rank&      operator++();
  const Rank operator++(int) const;

 private:
  mpi_rank m_rank{MPI_PROC_NULL};
};

Rank operator+(Rank const& lhs, Rank const& rhs) noexcept;
Rank operator-(Rank const& lhs, Rank const& rhs) noexcept;
Rank operator^(Rank const& lhs, Rank const& rhs) RTLX_NOEXCEPT;
Rank operator%(Rank const& lhs, Rank const& rhs) noexcept;

bool operator==(Rank const& lhs, Rank const& rhs) noexcept;
bool operator!=(Rank const& lhs, Rank const& rhs) noexcept;
bool operator>(Rank const& lhs, Rank const& rhs) noexcept;
bool operator<(Rank const& lhs, Rank const& rhs) noexcept;

std::ostream& operator<<(std::ostream& os, Rank const& p);

class Context {
 public:
  using size_type = std::uint32_t;

 public:
  explicit Context(MPI_Comm comm);

  Rank rank() const noexcept;

  size_type size() const noexcept;

  MPI_Comm mpiComm() const noexcept;

 private:
  size_type m_size{};
  Rank      m_rank{};
  MPI_Comm  m_comm{MPI_COMM_NULL};
};

Context splitSharedComm(Context const& baseComm);

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
