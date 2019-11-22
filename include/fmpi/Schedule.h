#ifndef FMPI_SCHEDULE_H
#define FMPI_SCHEDULE_H

#include <fmpi/mpi/Mpi.h>

namespace fmpi {

class FlatHandshake {
 public:
  static constexpr const char* NAME = "Ring";

  auto sendRank(mpi::Context const& comm, uint32_t phase) const -> mpi::Rank;

  auto recvRank(mpi::Context const& comm, uint32_t phase) const -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;

 private:
  static auto hypercube(mpi::Context const& comm, uint32_t phase)  -> mpi::Rank;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  auto sendRank(mpi::Context const& comm, uint32_t phase) const -> mpi::Rank;

  auto recvRank(mpi::Context const& comm, uint32_t phase) const -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;

 private:
  static auto factor_even(mpi::Context const& comm, uint32_t phase)  -> mpi::Rank;

  static auto factor_odd(mpi::Context const& comm, uint32_t phase)  -> mpi::Rank;
};

class Linear {
 public:
  static constexpr const char* NAME = "Linear";

  static auto sendRank(mpi::Context const& comm, uint32_t phase) noexcept -> mpi::Rank;

  static auto recvRank(mpi::Context const& comm, uint32_t phase) noexcept -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;
};

}  // namespace fmpi

#endif
