#include <fmpi/Math.h>
#include <fmpi/Schedule.h>

namespace fmpi {

mpi::rank_t FlatHandshake::sendRank(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  return isPow2<unsigned>(comm.size())
             ? hypercube(comm, phase)
             : mod(comm.rank() + phase, comm.size());
}

mpi::rank_t FlatHandshake::recvRank(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  return isPow2<unsigned>(comm.size())
             ? hypercube(comm, phase)
             : mod(comm.rank() - phase, comm.size());
}

mpi::rank_t FlatHandshake::hypercube(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  RTLX_ASSERT(isPow2<unsigned>(comm.size()));
  using unsigned_t = std::make_unsigned<mpi::rank_t>::type;
  return unsigned_t(comm.rank()) ^ unsigned_t(phase);
}

mpi::rank_t OneFactor::sendRank(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  return (comm.size() % 2) != 0 ? factor_odd(comm, phase) : factor_even(comm, phase);
}

mpi::rank_t OneFactor::recvRank(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  return sendRank(comm, phase);
}

mpi::rank_t OneFactor::factor_even(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  mpi::rank_t idle = mod(comm.size() * phase / 2, comm.size() - 1);

  if (comm.rank() == comm.size() - 1) {
    return idle;
  }

  if (comm.rank() == idle) {
    return comm.size() - 1;
  }

  return mod(phase - comm.rank(), comm.size() - 1);
}

mpi::rank_t OneFactor::factor_odd(
    mpi::MpiCommCtx const& comm, mpi::rank_t phase) const noexcept
{
  return mod(phase - comm.rank(), comm.size());
}
}  // namespace fmpi
