#include <mpi.h>

#include <fmpi/Debug.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/Enum.hpp>

namespace mpi {

static MPI_Comm comm_world = MPI_COMM_NULL;

static bool initialized = false;

Context::Context(MPI_Comm comm)
  : m_comm(comm) {
  int sz;

  int rank;
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &sz), MPI_SUCCESS);
  m_size = sz;

  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &rank), MPI_SUCCESS);
  m_rank = Rank{rank};
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

Context const& Context::world() {
  static Context ctx{comm_world};
  return ctx;
}

bool is_thread_main() {
  int flag;
  FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));
  return flag == 1;
}

bool initialize(int* argc, char*** argv, ThreadLevel level) {
  if (initialized) return true;

  auto const required = rtlx::to_underlying(level);
  int        provided;

  auto const init = MPI_Init_thread(argc, argv, required, &provided);

  auto const dup = MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);

  auto const success = init == MPI_SUCCESS && dup == MPI_SUCCESS;

  initialized = success && required <= provided;

  return initialized;
}

void finalize() {
  RTLX_ASSERT_RETURNS(MPI_Comm_free(&comm_world), MPI_SUCCESS);
  RTLX_ASSERT_RETURNS(MPI_Finalize(), MPI_SUCCESS);
}
}  // namespace mpi
