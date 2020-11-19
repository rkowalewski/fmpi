#include <mpi.h>

#include <fmpi/Debug.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <limits>
#include <rtlx/Enum.hpp>

namespace mpi {

static MPI_Comm comm_world = MPI_COMM_NULL;

static bool          initialized = false;
static int32_t const mpi_tag_ub  = 32767;

Context::Context(MPI_Comm comm, bool free_self)
  : m_comm(comm)
  , m_collective_tag(mpi_tag_ub)
  , m_free_self(free_self) {
  int sz = 0;

  int32_t flag = 0;
  // Assert that MPI is initialized
  FMPI_CHECK_MPI(MPI_Initialized(&flag));
  FMPI_ASSERT(flag);

  // Rank
  int rank = 0;
  FMPI_CHECK_MPI(MPI_Comm_rank(m_comm, &rank));
  m_rank = Rank{rank};

  FMPI_CHECK_MPI(MPI_Comm_group(m_comm, &m_group));

  // Size
  FMPI_CHECK_MPI(MPI_Comm_size(m_comm, &sz));
  m_size = sz;
}

Context::Context(MPI_Comm comm)
  : Context(comm, true) {
}

void Context::abort(int error) const {
  MPI_Abort(m_comm, error);
}

int32_t Context::requestTagSpace(int32_t n) const {
  return m_collective_tag.fetch_sub(n);
  // auto ret = m_collective_tag--;
  // if (m_collective_tag == -1) {
  //  m_collective_tag = MPI_TAG_UB;
  //}
  // return ret;
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

#if 0
  // Maximum tag value
  int32_t flag = 0;

  int32_t tag_ub = 0;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag);

  // For whatever reason, OpenMPI provides different values on different ranks
  // although this does not conform with the MPI-3 standard. See
  // section 8.1.2.
  if (flag != 0) {
    MPI_Allreduce(&tag_ub, &mpi_tag_ub, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
#endif

  return initialized;
}

void finalize() {
  FMPI_CHECK_MPI(MPI_Comm_free(&comm_world));
  FMPI_CHECK_MPI(MPI_Finalize());
}
}  // namespace mpi
