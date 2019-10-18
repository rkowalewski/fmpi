#ifndef RTLX_COLL_SHMEM_H
#define RTLX_COLL_SHMEM_H

#include <cstdlib>

#include <rtlx/Assert.h>

#include <fmpi/Math.h>
#include <fmpi/mpi/Mpi.h>

#include <tlx/math/integer_log2.hpp>
#include <tlx/simple_vector.hpp>

#include <morton.h>

namespace fmpi {

enum class AllToAllAlgorithm;

template <class T, class Op>
inline void all2allMortonZSource(
    mpi::Context const&    ctx,
    mpi::ShmSegment<T> const& from,
    mpi::ShmSegment<T>&       to,
    int                       blocksize,
    Op&&                      op)
{
  using value_type = T;
  using iterator   = typename mpi::ShmSegment<T>::pointer;

  using unsigned_rank_t = typename std::make_unsigned<mpi::rank_t>::type;
  using unsigned_diff_t =
      typename std::make_unsigned<mpi::difference_type>::type;

  using morton_coords_t = std::pair<uint_fast32_t, uint_fast32_t>;

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::string s;
  if (rtlx::TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "All2AllMortonZSource";
    s = os.str();
  }

  auto trace = rtlx::TimeTrace{me, s};
  trace.tick(COMMUNICATION);

  RTLX_ASSERT(isPow2(static_cast<unsigned>(nr)));

  auto const log2 = tlx::integer_log2_floor(static_cast<unsigned_rank_t>(nr));
  // We want to guarantee that we do not only have a power of 2.
  // But we need a square as well.
  // RTLX_ASSERT((log2 % 2 == 0) || (nr == 2));

  char          rflag;
  const char    sflag      = 1;
  constexpr int notify_tag = 201;

  constexpr int maxReqs = 32;

  std::array<MPI_Request, maxReqs> reqs;
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  // round down to next even number
  unsigned_rank_t mask    = 0x1;
  auto const      ystride = log2 & ~mask;

  auto const xstride = (log2 & mask) ? ystride * 2 : ystride;

  std::size_t nreq = 0;
  for (mpi::rank_t y = 0; y < nr; y += ystride) {
    auto const x = me;

    auto const p = libmorton::morton2D_64_encode(x, y);

    auto const src = p / nr;

    FMPI_DBG_STREAM("point (" << y << "," << x << "), recv from: " << src);

    if (static_cast<mpi::rank_t>(src) != me) {
      std::cerr << "hello\n";
      RTLX_ASSERT_RETURNS(
          MPI_Irecv(
              &rflag,
              0,
              MPI_BYTE,
              src,
              notify_tag,
              ctx.mpiComm(),
              &reqs[nreq]),
          MPI_SUCCESS);

      ++nreq;
    }
  }

  // eventually we should allocate a small buffer with MPI

  using simple_vector =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  std::vector<std::pair<iterator, iterator>> chunks(nr / ystride);
  simple_vector                              rbuf(nr * blocksize);

  trace.tock(COMMUNICATION);

  std::size_t firstChunk = me * nr;
  std::size_t lastChunk  = firstChunk + nr;

  for (std::size_t morton = firstChunk; morton < lastChunk; ++morton) {
    trace.tick(COMMUNICATION);
    morton_coords_t coords{};
    libmorton::morton2D_64_decode(morton, coords.second, coords.first);

    unsigned_rank_t srcRank, dstRank;
    unsigned_diff_t srcOffset, dstOffset;

    std::tie(srcRank, srcOffset) = coords;

    std::swap(coords.first, coords.second);

    std::tie(dstRank, dstOffset) = coords;

    auto srcAddr = std::next(from.base(srcRank), srcOffset * blocksize);
    auto ymask   = ystride - 1;
    auto xmask   = xstride - 1;
    auto col     = srcOffset & xmask;
    auto row     = srcRank & ymask;

    auto block     = col * ystride * blocksize;
    auto blockOffs = row * blocksize;

    auto buf = std::next(rbuf.begin(), block + blockOffs);
    // auto dstAddr = std::next(to.base(dstRank), dstOffset * blocksize);

    FMPI_DBG_STREAM("copy to offset: " << std::distance(rbuf.begin(), buf));
    // std::memcpy(dstAddr, srcAddr, blocksize * sizeof(value_type));
    std::copy(srcAddr, srcAddr + blocksize, buf);

    trace.tock(COMMUNICATION);

    if (row == ymask) {
      auto range = fmpi::range<unsigned>(row - ymask, row + 1);

      trace.tick(MERGE);
      std::transform(
          std::begin(range),
          std::end(range),
          std::begin(chunks),
          [buf = rbuf.begin(), chunksize = blocksize, block](auto offset) {
            auto offs = block + offset * chunksize;
            auto f    = std::next(buf, offs);
            auto l    = std::next(f, chunksize);
            return std::make_pair(f, l);
          });

      auto mergeBlock = (srcRank & ~ymask);
      auto dstAddr    = std::next(to.base(dstRank), mergeBlock * blocksize);

      FMPI_DBG_STREAM("merging to offset: " << mergeBlock * blocksize);

      op(chunks, dstAddr);
      trace.tock(MERGE);

      trace.tick(COMMUNICATION);
      if (static_cast<mpi::rank_t>(dstRank) != me) {
        FMPI_DBG_STREAM(
            "point (" << srcRank << "," << srcOffset
                      << "), send to: " << dstRank);

        RTLX_ASSERT_RETURNS(
            MPI_Send(
                &sflag,
                0,
                MPI_BYTE,
                dstRank,
                notify_tag,
                ctx.mpiComm()),
            MPI_SUCCESS);
      }
      trace.tock(COMMUNICATION);
    }
  }

  trace.tick(COMMUNICATION);

  RTLX_ASSERT_RETURNS(
      MPI_Waitall(nreq, &reqs[0], MPI_STATUSES_IGNORE), MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  auto range = fmpi::range<unsigned>(0, nr * blocksize, ystride * blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::begin(chunks),
      [buf       = to.base(ctx.rank()),
       chunksize = ystride * blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, chunksize);

        RTLX_ASSERT(std::is_sorted(f, l));

        return std::make_pair(f, l);
      });

  op(chunks, rbuf.begin());

  std::move(rbuf.data(), rbuf.data() + nr * blocksize, to.base(ctx.rank()));

  trace.tock(MERGE);

  RTLX_ASSERT(std::is_sorted(
      to.base(ctx.rank()), to.base(ctx.rank()) + nr * blocksize));
}

template <class T, class Op>
inline void all2allMortonZDest(
    mpi::Context const&    ctx,
    mpi::ShmSegment<T> const& from,
    mpi::ShmSegment<T>&       to,
    int                       blocksize,
    Op&&                      op)
{
  using value_type = T;
  using iterator   = typename mpi::ShmSegment<T>::pointer;

  using unsigned_rank_t = typename std::make_unsigned<mpi::rank_t>::type;
  using unsigned_diff_t =
      typename std::make_unsigned<mpi::difference_type>::type;

  using morton_coords_t = std::pair<uint_fast32_t, uint_fast32_t>;

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::string s;
  if (rtlx::TraceStore::GetInstance().enabled()) {
    std::ostringstream os;
    os << "All2AllMortonDest";
    s = os.str();
  }

  auto trace = rtlx::TimeTrace{me, s};
  trace.tick(COMMUNICATION);

  RTLX_ASSERT(isPow2(static_cast<unsigned>(nr)));

  auto const log2 = tlx::integer_log2_floor(static_cast<unsigned_rank_t>(nr));
  // We want to guarantee that we do not only have a power of 2.
  // But we need a square as well.
  // RTLX_ASSERT((log2 % 2 == 0) || (nr == 2));

  char          rflag;
  const char    sflag      = 1;
  constexpr int notify_tag = 201;

  constexpr int maxReqs = 32;

  std::array<MPI_Request, maxReqs> reqs;
  std::uninitialized_fill(std::begin(reqs), std::end(reqs), MPI_REQUEST_NULL);

  // round down to next even number
  unsigned_rank_t mask    = 0x1;
  auto const      ystride = log2 & ~mask;

  auto const xstride = (log2 & mask) ? ystride * 2 : ystride;

  mpi::rank_t nreq = 0;

  for (uint_fast32_t x = 0; x < static_cast<uint_fast32_t>(nr);
       x += xstride) {
    auto code =
        libmorton::morton2D_64_encode(x, static_cast<uint_fast32_t>(me));

    auto piece = code / nr;

    if (static_cast<mpi::rank_t>(piece) != me) {
      FMPI_DBG_STREAM(
          "point (" << me << "," << x << "), recv from: " << piece);
      RTLX_ASSERT_RETURNS(
          MPI_Irecv(
              &rflag,
              0,
              MPI_BYTE,
              piece,
              notify_tag,
              ctx.mpiComm(),
              &reqs[nreq]),
          MPI_SUCCESS);
    }
  }

  // eventually we should allocate a small buffer with MPI

  using simple_vector =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  simple_vector rbuf(nr * blocksize);

  std::size_t firstChunk = me * nr;
  std::size_t lastChunk  = firstChunk + nr;

  morton_coords_t coords{};
  libmorton::morton2D_64_decode(firstChunk, coords.second, coords.first);

  auto const firstX = coords.second;
  auto const firstY = coords.first;

  trace.tock(COMMUNICATION);

  for (std::size_t morton = firstChunk; morton < lastChunk; ++morton) {
    trace.tick(COMMUNICATION);
    libmorton::morton2D_64_decode(morton, coords.second, coords.first);

    unsigned_rank_t srcRank, dstRank;
    unsigned_diff_t srcOffset, dstOffset;

    std::tie(dstRank, dstOffset) = coords;

    std::swap(coords.first, coords.second);

    std::tie(srcRank, srcOffset) = coords;

    auto srcAddr  = std::next(from.base(srcRank), srcOffset * blocksize);
    auto ymask    = ystride - 1;
    auto xmask    = xstride - 1;
    auto blockCol = dstOffset & xmask;
    auto blockRow = dstRank & ymask;

    auto block     = blockRow * xstride * blocksize;
    auto blockOffs = blockCol * blocksize;

    auto buf = std::next(rbuf.begin(), block + blockOffs);

    FMPI_DBG_STREAM("copy to offset: " << std::distance(rbuf.begin(), buf));

    std::copy(srcAddr, srcAddr + blocksize, buf);

    trace.tock(COMMUNICATION);
  }

  std::vector<std::pair<iterator, iterator>> chunks(
      std::max(ystride, xstride));

  for (auto dstRank = firstY; dstRank < firstY + ystride; ++dstRank) {
    trace.tick(MERGE);
    auto range = fmpi::range<std::size_t>(0, xstride * blocksize, blocksize);

    auto block = (dstRank % ystride) * xstride * blocksize;

    std::transform(
        std::begin(range),
        std::end(range),
        std::begin(chunks),
        [buf = rbuf.begin(), block, chunksize = blocksize](auto offset) {
          auto dst = buf + block;
          auto f   = std::next(dst, offset);
          auto l   = std::next(f, chunksize);
          FMPI_DBG_STREAM(
              "merging pair (" << std::distance(buf, f) << ", "
                               << std::distance(buf, l) << ")");
          return std::make_pair(f, l);
        });

    auto dstAddr = std::next(to.base(dstRank), firstX * blocksize);

    FMPI_DBG_STREAM("merging to offset: " << firstX * blocksize);

    op(chunks, dstAddr);
    trace.tock(MERGE);

    if (static_cast<mpi::rank_t>(dstRank) != me) {
      trace.tick(COMMUNICATION);
      FMPI_DBG_STREAM("notify rank: " << dstRank);

      RTLX_ASSERT_RETURNS(
          MPI_Send(
              &sflag, 0, MPI_BYTE, dstRank, notify_tag, ctx.mpiComm()),
          MPI_SUCCESS);
      trace.tock(COMMUNICATION);
    }
  }

  trace.tick(COMMUNICATION);

  RTLX_ASSERT_RETURNS(
      MPI_Waitall(nreq, &reqs[0], MPI_STATUSES_IGNORE), MPI_SUCCESS);

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  auto stride = xstride * blocksize;
  auto range  = fmpi::range<unsigned>(0, nr * blocksize, stride);

  chunks.resize(std::min(xstride, ystride));

  std::transform(
      std::begin(range),
      std::end(range),
      std::begin(chunks),
      [buf = to.base(ctx.rank()), chunksize = stride](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, chunksize);

        RTLX_ASSERT(std::is_sorted(f, l));

        return std::make_pair(f, l);
      });

  op(chunks, rbuf.begin());

  std::move(rbuf.data(), rbuf.data() + nr * blocksize, to.base(ctx.rank()));

  trace.tock(MERGE);

  RTLX_ASSERT(std::is_sorted(
      to.base(ctx.rank()), to.base(ctx.rank()) + nr * blocksize));
}

template <class T, class Op>
inline void all2allNaive(
    mpi::Context const&    ctx,
    mpi::ShmSegment<T> const& from,
    mpi::ShmSegment<T>&       to,
    int                       blocksize,
    Op&&                      op)
{
  using value_type = T;
  using iterator   = typename mpi::ShmSegment<T>::pointer;

  auto nr = ctx.size();
  auto me = ctx.rank();

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  auto trace = rtlx::TimeTrace{me, "All2AllNaive"};

  trace.tick(COMMUNICATION);

  for (mpi::rank_t i = 0; i < nr; ++i) {
    auto srcAddr = std::next(from.base(i), me * blocksize);
    auto dstAddr = std::next(rbuf.get(), i * blocksize);

    std::copy(srcAddr, std::next(srcAddr, blocksize), dstAddr);
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<iterator, iterator>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = rbuf.get(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, to.base(me));

  trace.tock(MERGE);

  RTLX_ASSERT(std::is_sorted(to.base(me), to.base(me) + nr * blocksize));
}
}  // namespace fmpi

#endif
