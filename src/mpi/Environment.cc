#include <mpi.h>

#include <fmpi/mpi/Environment.hpp>
#include <rtlx/Enum.hpp>

namespace mpi {

static MPI_Comm comm_world = MPI_COMM_NULL;

static bool initialized = false;

Context::Context(MPI_Comm comm, bool free_self)
  : m_comm(comm)
  , m_free_self(free_self) {
  int sz = 0;

  int rank = 0;
  FMPI_CHECK_MPI(MPI_Comm_size(m_comm, &sz));
  m_size = sz;

  FMPI_CHECK_MPI(MPI_Comm_rank(m_comm, &rank));
  m_rank = Rank{rank};
}

Context::Context(MPI_Comm comm)
  : Context(comm, false) {
}

Context::~Context() {
  if (m_free_self) {
    FMPI_CHECK_MPI(MPI_Comm_free(&m_comm));
  }
}

Context splitSharedComm(Context const& baseComm) {
  MPI_Comm sharedComm = MPI_COMM_NULL;
  // split world into shared memory communicator
  FMPI_CHECK_MPI(MPI_Comm_split_type(
      baseComm.mpiComm(),
      MPI_COMM_TYPE_SHARED,
      0,
      MPI_INFO_NULL,
      &sharedComm));

  return Context{sharedComm, true};
}

Context const& Context::world() {
  static Context ctx{comm_world, false};
  return ctx;
}

bool is_thread_main() {
  int flag = 0;
  FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));
  return flag == 1;
}

bool initialize(int* argc, char*** argv, ThreadLevel level) {
  if (initialized) {
    return true;
  }

  auto const required = rtlx::to_underlying(level);
  int        provided = 0;

  auto const init = MPI_Init_thread(argc, argv, required, &provided);

  auto const dup = MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);

  auto const success = init == MPI_SUCCESS && dup == MPI_SUCCESS;

  initialized = success && required <= provided;

  return initialized;
}

void finalize() {
  FMPI_CHECK_MPI(MPI_Comm_free(&comm_world));
  FMPI_CHECK_MPI(MPI_Finalize());
}
}  // namespace mpi
