#ifndef MPI__ENVIRONMENT_H
#define MPI__ENVIRONMENT_H

#include <mpi.h>

#include <fmpi/mpi/TypeMapper.h>

namespace mpi {

using mpi_rank = int32_t;

struct Rank {
 public:
  Rank() = default;
  explicit Rank(mpi_rank rank) noexcept;
  operator mpi_rank() const noexcept;

 private:
  mpi_rank m_rank{MPI_PROC_NULL};
};

bool operator==(Rank lhs, Rank rhs) noexcept;
bool operator!=(Rank lhs, Rank rhs) noexcept;

class MpiCommCtx {
 public:
  using size_type = std::uint32_t;

 public:
  explicit MpiCommCtx(MPI_Comm comm);

  Rank rank() const noexcept;

  size_type size() const noexcept;

  MPI_Comm mpiComm() const noexcept;

 private:
  size_type m_size{};
  Rank      m_rank{};
  MPI_Comm  m_comm{MPI_COMM_NULL};
};

MpiCommCtx splitSharedComm(MpiCommCtx const& baseComm);

}  // namespace mpi

#endif
