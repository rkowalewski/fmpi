#ifndef FMPI_MPI_ENVIRONMENT_H
#define FMPI_MPI_ENVIRONMENT_H

#include <cstdint>

#include <mpi.h>

#include <fmpi/mpi/TypeMapper.h>

namespace mpi {

using mpi_rank = int32_t;

struct Rank {
 public:
  Rank() = default;
  explicit Rank(mpi_rank rank) noexcept;
  operator mpi_rank() const noexcept;  // NOLINT

  Rank&      operator++();
  const Rank operator++(int) const;

 private:
  mpi_rank m_rank{MPI_PROC_NULL};
};

bool operator==(Rank lhs, Rank rhs) noexcept;
bool operator!=(Rank lhs, Rank rhs) noexcept;

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

#endif
