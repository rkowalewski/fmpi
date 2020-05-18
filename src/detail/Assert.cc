#include <fmpi/detail/Assert.hpp>

#include <cstdio>
#include <cstdlib>

#include <fmpi/mpi/Environment.hpp>

void fmpi::detail::handle_failed_assert(
    const char* msg, const char* file, int line, const char* fnc) noexcept {

  auto const& world = mpi::Context::world();
  constexpr int errc = 1;

  std::fprintf(
      stderr,
      "[%s, Rank %d] Assertion failure in function %s (%s:%d): %s.\n",
      FMPI_LOG_PREFIX,
      static_cast<int>(world.rank()),
      fnc,
      file,
      line,
      msg);

  MPI_Abort(world.mpiComm(), errc);
}

void fmpi::detail::handle_warning(
    const char* msg, const char* file, int line, const char* fnc) noexcept {
  auto const& world = mpi::Context::world();
  std::fprintf(
      stderr,
      "[%s, Rank %d] Warning triggered in function %s (%s:%d): %s.\n",
      FMPI_LOG_PREFIX,
      static_cast<int>(world.rank()),
      fnc,
      file,
      line,
      msg);
}
