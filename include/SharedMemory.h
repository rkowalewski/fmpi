#ifndef A2A_COLL_SHMEM_H
#define A2A_COLL_SHMEM_H

#include <cstdlib>

#include <mpi.h>

namespace a2a {

enum class AllToAllAlgorithm;

template <
    AllToAllAlgorithm algo,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void scatteredPairwiseWaitsome(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&& op)
{
}
}  // namespace a2a

#endif
