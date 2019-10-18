#include <fmpi/Math.h>
#include <fmpi/Schedule.h>

namespace fmpi {

using namespace mpi;

Rank FlatHandshake::sendRank(
    mpi::MpiCommCtx const& comm, mpi_rank phase) const noexcept
{
  auto r = isPow2(comm.size())
               ? hypercube(comm, phase)
               : mod(comm.rank() + phase, static_cast<mpi_rank>(comm.size()));
  return Rank{r};
}

Rank FlatHandshake::recvRank(
    mpi::MpiCommCtx const& comm, mpi_rank phase) const noexcept
{
  auto r = isPow2(comm.size())
               ? hypercube(comm, phase)
               : mod(comm.rank() - phase, static_cast<mpi_rank>(comm.size()));
  return Rank{r};
}

Rank FlatHandshake::hypercube(
    mpi::MpiCommCtx const& comm, mpi_rank phase) const noexcept
{
  RTLX_ASSERT(isPow2(comm.size()));
  return Rank{comm.rank() ^ phase};
}

Rank OneFactor::sendRank(mpi::MpiCommCtx const& comm, mpi_rank phase) const
    noexcept
{
  return (comm.size() % 2) != 0 ? factor_odd(comm, phase)
                                : factor_even(comm, phase);
}

Rank OneFactor::recvRank(mpi::MpiCommCtx const& comm, mpi_rank phase) const
    noexcept
{
  return sendRank(comm, phase);
}

Rank OneFactor::factor_even(mpi::MpiCommCtx const& comm, mpi_rank phase) const
    noexcept
{
  Rank idle = static_cast<Rank>(
      mod<mpi_rank>(comm.size() * phase / 2, comm.size() - 1));

  if (comm.rank() == static_cast<mpi_rank>(comm.size()) - 1) {
    return idle;
  }

  if (comm.rank() == idle) {
    return Rank{static_cast<mpi_rank>(comm.size()) - 1};
  }

  return Rank{mod<mpi_rank>(phase - comm.rank(), comm.size() - 1)};
}

Rank OneFactor::factor_odd(mpi::MpiCommCtx const& comm, mpi_rank phase) const
    noexcept
{
  return Rank{mod<mpi_rank>(phase - comm.rank(), comm.size())};
}
}  // namespace fmpi
