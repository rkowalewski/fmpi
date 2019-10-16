#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>

namespace mpi {

MpiCommCtx::MpiCommCtx(MPI_Comm const& base)
{
  m_comm = base;
  _initialize();
}

MpiCommCtx::MpiCommCtx(MPI_Comm&& base)
{
  m_comm = std::move(base);
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
}

void MpiCommCtx::_initialize()
{
  RTLX_ASSERT_RETURNS(MPI_Comm_size(m_comm, &m_size), MPI_SUCCESS);
  RTLX_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &m_rank), MPI_SUCCESS);
}

MpiCommCtx splitSharedComm(MpiCommCtx const& baseComm)
{
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

  return MpiCommCtx{sharedComm};
}

}  // namespace mpi
