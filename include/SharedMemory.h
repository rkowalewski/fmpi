#ifndef A2A_COLL_SHMEM_H
#define A2A_COLL_SHMEM_H

#include <cstdlib>

#include <Debug.h>
#include <Mpi.h>

#include <tlx/math/integer_log2.hpp>
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

  using morton_coords_t = std::pair<uint_fast32_t, uint_fast32_t>;

  auto nr = from.ctx().size();
  auto me = from.ctx().rank();

  std::string s;
  if (TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "All2AllMorton";
    s = os.str();
  }

  auto trace = TimeTrace{me, s};
  trace.tick(COMMUNICATION);

  A2A_ASSERT(from.ctx().mpiComm() == to.ctx().mpiComm());
  A2A_ASSERT(isPow2(static_cast<unsigned>(nr)));

  unsigned const log2 = tlx::integer_log2_floor(static_cast<unsigned>(nr));
  // We want to guarantee that we do not only have a power of 2.
  // But we need a square as well.
  // A2A_ASSERT((log2 % 2 == 0) || (nr == 2));

  char          rflag;
  const char    sflag      = 1;
  constexpr int notify_tag = 201;

  // post asynchron receives
  // std::size_t supporters = nr / 2;

  constexpr int maxReqs = 32;

  std::array<MPI_Request, maxReqs> reqs;
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  // auto mask = std::numeric_limits<decltype(log2)>::max() << 1;

  // round down to next even number
  auto const ystride = (log2 % 2) ? log2 - 1 : log2;

  std::size_t nreq = 0;
  for (mpi::rank_t y = 0; y < nr; y += ystride) {
    auto const x = me;

    auto const p = libmorton::morton2D_64_encode(x, y);

    auto const src = p / nr;

    P(me << " point (" << y << "," << x << "), recv from: " << src);

    if (static_cast<mpi::rank_t>(src) != me) {
      A2A_ASSERT_RETURNS(
          MPI_Irecv(
              &rflag,
              0,
              MPI_BYTE,
              src,
              notify_tag,
              from.ctx().mpiComm(),
              &reqs[nreq]),
          MPI_SUCCESS);

      ++nreq;
    }
  }

  // eventually we should allocate a small buffer with MPI

  std::size_t firstChunk = me * nr;
  std::size_t lastChunk  = firstChunk + nr;

  using simple_vector =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  std::vector<std::pair<iterator, iterator>> chunks(ystride);
  simple_vector                              rbuf(nr * blocksize);

  //#pragma omp parallel
  for (std::size_t chunk = firstChunk; chunk < lastChunk; ++chunk) {
    morton_coords_t coords{};
    libmorton::morton2D_64_decode(chunk, coords.second, coords.first);

    using unsigned_rank_t = typename std::make_unsigned<mpi::rank_t>::type;
    using unsigned_diff_t =
        typename std::make_unsigned<mpi::difference_type>::type;

    unsigned_rank_t srcRank, dstRank;
    unsigned_diff_t srcOffset, dstOffset;

    std::tie(srcRank, srcOffset) = coords;

    std::swap(coords.first, coords.second);

    std::tie(dstRank, dstOffset) = coords;

    auto srcAddr = std::next(from.base(srcRank), srcOffset * blocksize);
    auto mask    = ystride - 1;
    auto col     = srcOffset & mask;
    auto row     = srcRank & mask;

    auto block     = col * ystride * blocksize;
    auto blockOffs = row * blocksize;

    auto dstAddr = std::next(rbuf.begin(), block + blockOffs);
    // auto dstAddr = std::next(to.base(dstRank), dstOffset * blocksize);

    P(me << " copy to offset: " << std::distance(rbuf.begin(), dstAddr));
    // std::memcpy(dstAddr, srcAddr, blocksize * sizeof(value_type));
    std::copy(srcAddr, srcAddr + blocksize, dstAddr);

    if (row == mask) {
      auto range = a2a::range<unsigned>(row - mask, row + 1);

      std::transform(
          std::begin(range),
          std::end(range),
          std::begin(chunks),
          [buf = rbuf.begin(), chunksize = blocksize, me, block](
              auto offset) {
            auto offs = block + offset * chunksize;
            auto f    = std::next(buf, offs);
            auto l    = std::next(f, chunksize);
            P(me << " merging block: " << block << ", chunk: (" << offs
                 << ", " << offs + chunksize << ")");
            return std::make_pair(f, l);
          });

      auto shift     = tlx::integer_log2_floor(ystride);
      auto blockMask = std::numeric_limits<decltype(ystride)>::max() << shift;
      auto mergeBlock = (srcRank & blockMask);
      auto mergeDst   = std::next(to.base(dstRank), mergeBlock * blocksize);

      P(me << " merging to offset: " << mergeBlock * blocksize);

      op(chunks, mergeDst);

      if (static_cast<mpi::rank_t>(dstRank) != me) {
        P(me << " point (" << srcRank << "," << srcOffset
             << "), send to: " << dstRank);

        A2A_ASSERT_RETURNS(
            MPI_Send(
                &sflag,
                0,
                MPI_BYTE,
                dstRank,
                notify_tag,
                from.ctx().mpiComm()),
            MPI_SUCCESS);
      }
    }
  }

  A2A_ASSERT_RETURNS(
      MPI_Waitall(nreq, &reqs[0], MPI_STATUSES_IGNORE), MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  // TODO: This barrier can be eliminate if we signal destination ranks
  // after we finished the copy
  // MPI_Barrier(from.ctx().mpiComm());

#if 0
  int nCompleted = 0;

  std::array<MPI_Request, maxReqs> creqs = {MPI_REQUEST_NULL};

  while (nCompleted < nreq) {
    int nc;
    A2A_ASSERT_RETURNS(
        MPI_Waitsome(nreq, reqs, &nc, &(creqs[0]), MPI_STATUSES_IGNORE),
        MPI_SUCCESS);

    nCompleted += nc;

    if (nCompleted > 1) {

    }
  }
#endif
  auto range = a2a::range<unsigned>(0, nr * blocksize, ystride * blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::begin(chunks),
      [buf = to.base(), chunksize = ystride * blocksize, me](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, chunksize);

        A2A_ASSERT(std::is_sorted(f, l));

        P(me << " merging chunk: (" << offset << ", " << offset + chunksize
             << ")");
        return std::make_pair(f, l);
      });

  op(chunks, rbuf.begin());

  std::move(rbuf.data(), rbuf.data() + nr * blocksize, to.base());

  trace.tock(MERGE);

  A2A_ASSERT(std::is_sorted(to.base(), to.base() + nr * blocksize));
}
}  // namespace a2a

#endif
