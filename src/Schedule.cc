#include <fmpi/Math.h>
#include <fmpi/Schedule.h>

namespace fmpi {

using namespace mpi;

Rank FlatHandshake::sendRank(mpi::Context const& comm, uint32_t phase) const
{
  return isPow2(comm.size()) ? hypercube(comm, phase)
                             : mod(comm.rank() + static_cast<Rank>(phase),
                                   static_cast<Rank>(comm.size()));
}

Rank FlatHandshake::recvRank(mpi::Context const& comm, uint32_t phase) const
{
  auto r = isPow2(comm.size()) ? hypercube(comm, phase)
                               : mod(comm.rank() - static_cast<Rank>(phase),
                                     static_cast<Rank>(comm.size()));
  return Rank{r};
}

Rank FlatHandshake::hypercube(mpi::Context const& comm, uint32_t phase) const
{
  RTLX_ASSERT(isPow2(comm.size()));
  return comm.rank() ^ static_cast<Rank>(phase);
}

Rank OneFactor::sendRank(mpi::Context const& comm, uint32_t phase) const
{
  return (comm.size() % 2) != 0 ? factor_odd(comm, phase)
                                : factor_even(comm, phase);
}

Rank OneFactor::recvRank(mpi::Context const& comm, uint32_t phase) const
{
  return sendRank(comm, phase);
}

Rank OneFactor::factor_even(mpi::Context const& comm, uint32_t phase) const
{
  auto idle =
      mod(static_cast<Rank>(comm.size() * phase / 2),
          static_cast<Rank>(comm.size() - 1));

  if (comm.rank() == static_cast<Rank>(comm.size() - 1)) {
    return idle;
  }

  if (comm.rank() == idle) {
    return static_cast<Rank>(comm.size() - 1);
  }

  return mod(
      static_cast<Rank>(phase) - comm.rank(),
      static_cast<Rank>(comm.size() - 1));
}

Rank OneFactor::factor_odd(mpi::Context const& comm, uint32_t phase) const
{
  return mod(
      static_cast<Rank>(phase) - comm.rank(), static_cast<Rank>(comm.size()));
}

mpi::Rank Linear::sendRank(mpi::Context const& /* unused */, uint32_t phase) const noexcept {
  return static_cast<Rank>(phase);
}

mpi::Rank Linear::recvRank(mpi::Context const& /* unused */, uint32_t phase) const noexcept {
  return static_cast<Rank>(phase);
}

}  // namespace fmpi
