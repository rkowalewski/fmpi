#ifndef A2A_COLL_SHMEM_H
#define A2A_COLL_SHMEM_H

#include <cstdlib>

#include <Mpi.h>

#include <morton.h>

#include <Math.h>

namespace a2a {

enum class AllToAllAlgorithm;

template <class T, class Op>
inline void all2allMorton(
    mpi::ShmSegment<T> const& from,
    mpi::ShmSegment<T>&       to,
    int                       blocksize,
    Op&&                      op)
{
  A2A_ASSERT(from.ctx().mpiComm() == to.ctx().mpiComm());
  A2A_ASSERT(isPow2(static_cast<unsigned>(from.ctx().size())));

  (void)op;
  (void)blocksize;
}
}  // namespace a2a

#endif
