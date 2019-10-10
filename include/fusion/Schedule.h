#ifndef SCHEDULE_H
#define SCHEDULE_H

#include <fusion/mpi/Mpi.h>

namespace a2a {

enum class AllToAllAlgorithm { FLAT_HANDSHAKE, ONE_FACTOR };

class FlatHandshake {
 public:
  static constexpr const char* NAME = "FlatHandshake";

  mpi::rank_t sendRank(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
      noexcept;

  mpi::rank_t recvRank(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
      noexcept;

 private:
  mpi::rank_t hypercube(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
      noexcept;
};

class OneFactor {
 public:
  static constexpr const char* NAME = "OneFactor";

  mpi::rank_t sendRank(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
      noexcept;

  mpi::rank_t recvRank(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
      noexcept;

 private:
  mpi::rank_t factor_even(
      mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept;

  mpi::rank_t factor_odd(mpi::MpiCommCtx const& comm, mpi::rank_t phase) const
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
}  // namespace a2a

#endif
