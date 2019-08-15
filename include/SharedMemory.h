#ifndef A2A_COLL_SHMEM_H
#define A2A_COLL_SHMEM_H

#include <cstdlib>

#include <Mpi.h>

#include <Math.h>

namespace a2a {

enum class AllToAllAlgorithm;

template <
    AllToAllAlgorithm algo,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void all2allShmem(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, MPI_Win, Op&& op)
{
  mpi::rank_t nr, me;

  MPI_Comm_size(comm, &nr);
  MPI_Comm_rank(comm, &me);
  A2A_ASSERT(a2a::isPow2(static_cast<unsigned>(nr)));


}
}  // namespace a2a

#endif
