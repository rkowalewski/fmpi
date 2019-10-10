#include <fusion/mpi/Environment.h>

#include <Debug.h>

namespace mpi {

MpiCommCtx::MpiCommCtx(MPI_Comm const& base)
{
  A2A_ASSERT_RETURNS(MPI_Comm_dup(base, &m_comm), MPI_SUCCESS);
  _initialize();
}

MpiCommCtx::MpiCommCtx(MPI_Comm&& base)
{
  if (base == MPI_COMM_WORLD) {
    A2A_ASSERT_RETURNS(MPI_Comm_dup(base, &m_comm), MPI_SUCCESS);
  }
  else {
    m_comm = std::move(base);
  }
  _initialize();
}

rank_t MpiCommCtx::rank() const noexcept
{
  return m_rank;
}

rank_t MpiCommCtx::size() const noexcept
{
  return m_size;
}

MPI_Comm const& MpiCommCtx::mpiComm() const noexcept
{
  return m_comm;
}

MpiCommCtx::MpiCommCtx(MpiCommCtx&& other) noexcept
{
  *this = std::move(other);
}

MpiCommCtx& MpiCommCtx::operator=(MpiCommCtx&& other) noexcept
{
  std::swap(m_comm, other.m_comm);
  std::swap(m_size, other.m_size);
  std::swap(m_rank, other.m_rank);

  other.m_comm = MPI_COMM_NULL;
  return *this;
}

MpiCommCtx::~MpiCommCtx()
{
  if (m_comm != MPI_COMM_NULL && m_comm != MPI_COMM_WORLD) {
    A2A_ASSERT_RETURNS(MPI_Comm_free(&m_comm), MPI_SUCCESS);
  }
}

void MpiCommCtx::_initialize()
{
  A2A_ASSERT_RETURNS(MPI_Comm_size(m_comm, &m_size), MPI_SUCCESS);
  A2A_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &m_rank), MPI_SUCCESS);
}


MpiCommCtx splitSharedComm(MpiCommCtx const& baseComm)
{
  MPI_Comm sharedComm;
  // split world into shared memory communicator
  A2A_ASSERT_RETURNS(
      MPI_Comm_split_type(
          baseComm.mpiComm(),
          MPI_COMM_TYPE_SHARED,
          0,
          MPI_INFO_NULL,
          &sharedComm),
      MPI_SUCCESS);

  return MpiCommCtx{std::move(sharedComm)};
}

}  // namespace mpi
