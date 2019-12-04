#ifndef FMPI_SCHEDULE_HPP
#define FMPI_SCHEDULE_HPP

#include <fmpi/mpi/Environment.hpp>

namespace fmpi {

class FlatHandshake {
 public:
  static constexpr const char* NAME = "Ring";

  static auto sendRank(::mpi::Context const& comm, uint32_t phase)
      -> ::mpi::Rank;

  static auto recvRank(mpi::Context const& comm, uint32_t phase) -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;

 private:
  static auto hypercube(mpi::Context const& comm, uint32_t phase)
      -> mpi::Rank;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  static auto sendRank(mpi::Context const& comm, uint32_t phase) -> mpi::Rank;

  static auto recvRank(mpi::Context const& comm, uint32_t phase) -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;

 private:
  static auto factor_even(mpi::Context const& comm, uint32_t phase)
      -> mpi::Rank;

  static auto factor_odd(mpi::Context const& comm, uint32_t phase)
      -> mpi::Rank;
};

class Linear {
 public:
  static constexpr const char* NAME = "Linear";

  static auto sendRank(mpi::Context const& comm, uint32_t phase) noexcept
      -> mpi::Rank;

  static auto recvRank(mpi::Context const& comm, uint32_t phase) noexcept
      -> mpi::Rank;

  static auto phaseCount(mpi::Context const& comm) noexcept -> uint32_t;
};

}  // namespace fmpi

#endif
