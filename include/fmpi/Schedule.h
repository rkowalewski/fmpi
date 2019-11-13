#ifndef FMPI_SCHEDULE_H
#define FMPI_SCHEDULE_H

#include <fmpi/mpi/Mpi.h>

namespace fmpi {

class FlatHandshake {
 public:
  static constexpr const char* NAME = "Ring";

  mpi::Rank sendRank(mpi::Context const& comm, uint32_t phase) const;

  mpi::Rank recvRank(mpi::Context const& comm, uint32_t phase) const;

 private:
  mpi::Rank hypercube(mpi::Context const& comm, uint32_t phase) const;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  mpi::Rank sendRank(mpi::Context const& comm, uint32_t phase) const;

  mpi::Rank recvRank(mpi::Context const& comm, uint32_t phase) const;

 private:
  mpi::Rank factor_even(mpi::Context const& comm, uint32_t phase) const;

  mpi::Rank factor_odd(mpi::Context const& comm, uint32_t phase) const;
};

}  // namespace fmpi

#endif
