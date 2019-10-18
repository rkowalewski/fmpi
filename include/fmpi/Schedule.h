#ifndef SCHEDULE_H
#define SCHEDULE_H

#include <fmpi/mpi/Mpi.h>

namespace fmpi {

enum class AllToAllAlgorithm { FLAT_HANDSHAKE, ONE_FACTOR };

class FlatHandshake {
 public:
  static constexpr const char* NAME = "FlatHandshake";

  mpi::Rank sendRank(mpi::Context const& comm, mpi::mpi_rank phase) const;

  mpi::Rank recvRank(mpi::Context const& comm, mpi::mpi_rank phase) const;

 private:
  mpi::Rank hypercube(mpi::Context const& comm, mpi::mpi_rank phase) const;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  mpi::Rank sendRank(mpi::Context const& comm, mpi::mpi_rank phase) const;

  mpi::Rank recvRank(mpi::Context const& comm, mpi::mpi_rank phase) const;

 private:
  mpi::Rank factor_even(mpi::Context const& comm, mpi::mpi_rank phase) const;

  mpi::Rank factor_odd(mpi::Context const& comm, mpi::mpi_rank phase) const;
};

namespace detail {
template <AllToAllAlgorithm algo>
struct selectAlgorithm {
  using type = FlatHandshake;
};

template <>
struct selectAlgorithm<AllToAllAlgorithm::ONE_FACTOR> {
  using type = OneFactor;
};
}  // namespace detail
}  // namespace fmpi

#endif
