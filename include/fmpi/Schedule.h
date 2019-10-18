#ifndef SCHEDULE_H
#define SCHEDULE_H

#include <fmpi/mpi/Mpi.h>

namespace fmpi {

enum class AllToAllAlgorithm { FLAT_HANDSHAKE, ONE_FACTOR };

class FlatHandshake {
 public:
  static constexpr const char* NAME = "FlatHandshake";

  mpi::Rank sendRank(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;

  mpi::Rank recvRank(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;

 private:
  mpi::Rank hypercube(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  mpi::Rank sendRank(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;

  mpi::Rank recvRank(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;

 private:
  mpi::Rank factor_even(
      mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const noexcept;

  mpi::Rank factor_odd(mpi::MpiCommCtx const& comm, mpi::mpi_rank phase) const
      noexcept;
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
