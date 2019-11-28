#include <fmpi/Math.h>
#include <fmpi/Schedule.h>

using Rank    = mpi::Rank;
using Context = mpi::Context;

namespace fmpi {

auto FlatHandshake::sendRank(Context const& comm, uint32_t phase) -> ::Rank
{
  return isPow2(comm.size()) ? hypercube(comm, phase)
                             : mod(comm.rank() + static_cast<Rank>(phase),
                                   static_cast<Rank>(comm.size()));
}

auto FlatHandshake::recvRank(Context const& comm, uint32_t phase) -> Rank
{
  auto r = isPow2(comm.size()) ? hypercube(comm, phase)
                               : mod(comm.rank() - static_cast<Rank>(phase),
                                     static_cast<Rank>(comm.size()));
  return Rank{r};
}

auto FlatHandshake::hypercube(Context const& comm, uint32_t phase) -> Rank
{
  RTLX_ASSERT(isPow2(comm.size()));
  return comm.rank() ^ static_cast<Rank>(phase);
}

auto FlatHandshake::phaseCount(Context const& comm) noexcept -> uint32_t
{
  return comm.size();
}

auto OneFactor::sendRank(Context const& comm, uint32_t phase) -> Rank
{
  return (comm.size() % 2) != 0 ? factor_odd(comm, phase)
                                : factor_even(comm, phase);
}

auto OneFactor::recvRank(Context const& comm, uint32_t phase) -> Rank
{
  return sendRank(comm, phase);
}

auto OneFactor::factor_even(Context const& comm, uint32_t phase) -> Rank
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

auto OneFactor::factor_odd(Context const& comm, uint32_t phase) -> Rank
{
  return mod(
      static_cast<Rank>(phase) - comm.rank(), static_cast<Rank>(comm.size()));
}

auto OneFactor::phaseCount(Context const& comm) noexcept -> uint32_t
{
  return (comm.size() % 2) != 0U ? comm.size() : comm.size() - 1;
}

auto Linear::sendRank(Context const& /* unused */, uint32_t phase) noexcept
    -> Rank
{
  return static_cast<Rank>(phase);
}

auto Linear::recvRank(Context const& /* unused */, uint32_t phase) noexcept
    -> Rank
{
  return static_cast<Rank>(phase);
}

auto Linear::phaseCount(Context const& comm) noexcept -> uint32_t
{
  return comm.size();
}

}  // namespace fmpi
