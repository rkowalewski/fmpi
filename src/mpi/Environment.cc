#include <mpi.h>

#include <fmpi/Debug.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/Enum.hpp>

namespace mpi {

Context::Context(MPI_Comm comm)
  : m_comm(comm) {
  int sz;

  int rank;
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &sz), MPI_SUCCESS);
  m_size = sz;

  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &rank), MPI_SUCCESS);
  m_rank = Rank{rank};
}

auto Context::rank() const noexcept -> Rank {
  return m_rank;
}

auto Context::size() const noexcept -> Context::size_type {
  return m_size;
}

auto Context::mpiComm() const noexcept -> MPI_Comm {
  return m_comm;
}

auto splitSharedComm(Context const& baseComm) -> Context {
  MPI_Comm sharedComm;
  // split world into shared memory communicator
  RTLX_ASSERT_RETURNS(
      MPI_Comm_split_type(
          baseComm.mpiComm(),
          MPI_COMM_TYPE_SHARED,
          0,
          MPI_INFO_NULL,
          &sharedComm),
      MPI_SUCCESS);

  return Context{sharedComm};
}

Context& Context::world() {
  static Context ctx{MPI_COMM_WORLD};
  return ctx;
}

bool is_thread_main() {
  int flag;
  FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));
  return flag == 1;
}

bool initialize(int* argc, char*** argv, ThreadLevel level) {
  auto const required = rtlx::to_underlying(level);
  int        provided;

  auto const success = MPI_Init_thread(argc, argv, required, &provided);

  return success == MPI_SUCCESS && required <= provided;
}

void finalize() {
  RTLX_ASSERT_RETURNS(MPI_Finalize(), MPI_SUCCESS);
}
}  // namespace mpi
