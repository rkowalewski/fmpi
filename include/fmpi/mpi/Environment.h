#ifndef MPI__ENVIRONMENT_H
#define MPI__ENVIRONMENT_H

#include <fmpi/mpi/Types.h>

namespace mpi {

class MpiCommCtx {
 public:
  MpiCommCtx() = default;

  MpiCommCtx(MPI_Comm const& base);

  MpiCommCtx(MPI_Comm&& base);

  rank_t rank() const noexcept;

  rank_t size() const noexcept;

  MPI_Comm const& mpiComm() const noexcept;

  MpiCommCtx(MpiCommCtx&& other) noexcept;

  MpiCommCtx& operator=(MpiCommCtx&& other) noexcept;

  ~MpiCommCtx();

 private:
  void _initialize();

 private:
  MPI_Comm m_comm{MPI_COMM_NULL};
  rank_t   m_size{};
  rank_t   m_rank{};
};

MpiCommCtx splitSharedComm(MpiCommCtx const& baseComm);

}  // namespace mpi

#endif
