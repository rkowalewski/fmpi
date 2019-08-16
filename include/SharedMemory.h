#ifndef A2A_COLL_SHMEM_H
#define A2A_COLL_SHMEM_H

#include <cstdlib>

#include <Debug.h>
#include <Mpi.h>

#include <tlx/simple_vector.hpp>

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
  using value_type = T;
  using iterator   = typename mpi::ShmSegment<T>::pointer;

  auto nr = from.ctx().size();
  auto me = from.ctx().rank();

  A2A_ASSERT(from.ctx().mpiComm() == to.ctx().mpiComm());
  A2A_ASSERT(isPow2(static_cast<unsigned>(nr)));

  // eventually we should allocate a small buffer with MPI

  // We want to guarantee that we do not only have a power of 2.
  // But we need a square as well.
  A2A_ASSERT((static_cast<unsigned>(std::log2(nr)) % 2 == 0) || nr == 2);

  std::size_t firstChunk = me * nr;
  std::size_t lastChunk  = firstChunk + nr;

  using simple_vector =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  //#pragma omp parallel
  for (std::size_t chunk = firstChunk; chunk < lastChunk; ++chunk) {
    std::pair<uint_fast32_t, uint_fast32_t> coords{};
    libmorton::morton2D_64_decode(chunk, coords.second, coords.first);

    mpi::rank_t          srcRank, dstRank;
    mpi::difference_type srcOffset, dstOffset;

    std::tie(srcRank, srcOffset) = coords;

    std::swap(coords.first, coords.second);

    std::tie(dstRank, dstOffset) = coords;

    auto srcAddr = std::next(from.base(srcRank), srcOffset * blocksize);
    auto dstAddr = std::next(to.base(dstRank), dstOffset * blocksize);

    std::copy(srcAddr, srcAddr + blocksize, dstAddr);
  }

  simple_vector rbuf(nr * blocksize);

  std::vector<std::pair<iterator, iterator>> chunks;
  chunks.reserve(nr);

  auto range = a2a::range(0, nr * blocksize, blocksize);

  // TODO: This barrier can be eliminate if we signal destination ranks after
  // we finished the copy
  MPI_Barrier(from.ctx().mpiComm());

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = to.base(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, rbuf.data());

  std::move(rbuf.data(), rbuf.data() + nr * blocksize, to.base());

  A2A_ASSERT(std::is_sorted(to.base(), to.base() + nr * blocksize));
}
}  // namespace a2a

#endif
