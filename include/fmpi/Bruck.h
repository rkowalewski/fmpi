#ifndef BRUCK_H
#define BRUCK_H

#include <mpi.h>

#include <cmath>
#include <memory>

#include <rtlx/Assert.h>
#include <rtlx/Trace.h>

#include <fmpi/Constants.h>
#include <fmpi/Math.h>
#include <fmpi/mpi/Algorithm.h>
#include <fmpi/mpi/Environment.h>

#include <tlx/math/integer_log2.hpp>

namespace fmpi {

namespace detail {

template <class BidirIt, class OutputIt>
OutputIt reverse_copy_strided(
    BidirIt first, BidirIt last, std::size_t blocksize, OutputIt d_first)
{
  auto const n = std::distance(first, last);
  RTLX_ASSERT(n % blocksize == 0);

  auto const nb = n / blocksize;

  for (auto&& block : range(nb)) {
    std::copy(
        first + (nb - block - 1) * blocksize,
        first + (nb - block) * blocksize,
        d_first + block * blocksize);
  }

  return d_first + n;
}

static constexpr const char ROTATE[] = "Trotate";
static constexpr const char PACK[]   = "Tpack";
static constexpr const char UNPACK[] = "Tunpack";

}  // namespace detail

template <class InputIt, class OutputIt, class Op>
inline void bruck(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::TimeTrace{me, "Bruck"};

  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  trace.tick(detail::ROTATE);

  // O(p * blocksize)
  std::rotate_copy(
      begin,
      // n_first
      begin + me * blocksize,
      // last
      begin + blocksize * nr,
      // out
      out);

  trace.tock(detail::ROTATE);

  // Phase 2: Communication Rounds

  trace.tick(COMMUNICATION);

  // Reverse a buffer for send-recv exchanges
  // We never exchange more than (N/2) elements per round, so this buffer
  // suffices
  auto                       nels = size_t(nr) * blocksize;
  std::unique_ptr<value_t[]> tmpbuf{new value_t[nels]};

  auto* sendbuf = &tmpbuf[0];
  auto* recvbuf = &tmpbuf[nels / 2];

  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom, sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    // We exchange all blocks where the j-th bit is set

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

    trace.tick(detail::PACK);
    for (std::size_t block = 1; block < nr; ++block) {
      if (block & j) {
        std::copy(
            // begin
            out + block * blocksize,
            // end
            out + block * blocksize + blocksize,
            // tmp buf
            sendbuf + count * blocksize);
        ++count;
      }
    }
    trace.tock(detail::PACK);

    FMPI_CHECK(mpi::isend(
        sendbuf, blocksize * count, sendto, EXCH_TAG_BRUCK, ctx, &reqs[0]));

    FMPI_CHECK(mpi::irecv(
        recvbuf, blocksize * count, recvfrom, EXCH_TAG_BRUCK, ctx, &reqs[1]));

    FMPI_CHECK(mpi::waitall(reqs));

    trace.tick(detail::UNPACK);

    // c) unpack blocks into recv buffer
    count = 0;
    for (std::size_t block = 1; block < nr; ++block) {
      if (block & j) {
        std::copy(
            recvbuf + count * blocksize,
            recvbuf + count * blocksize + blocksize,
            out + block * blocksize);
        ++count;
      }
    }

    trace.tock(detail::UNPACK);
  }

#if 0
  // Phase 3: Process i rotates local elements by (i+1) blocks to the left in
  // a cyclic manner.
  std::rotate_copy(
      out,
      // n_first
      out + (me + 1) * blocksize,
      // last
      out + blocksize * nr,
      // out
      sendbuf);
#endif

  trace.tock(COMMUNICATION);
  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = out, blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, &tmpbuf[0]);

  std::move(&tmpbuf[0], &tmpbuf[nels], out);

  trace.tock(MERGE);
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_interleave(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;
  using simple_vector =
      tlx::SimpleVector<value_t, tlx::SimpleVectorMode::Normal>;

  std::vector<std::pair<InputIt, InputIt>> chunks;

  auto const    nels = size_t(nr) * blocksize;
  simple_vector buffer{nels};

  if (nr < 3) {
    chunks.emplace_back(
        std::make_pair(begin + me * blocksize, begin + (me + 1) * blocksize));

    if (nr == 1) {
      op(chunks, out);
      return;
    }

    auto other = static_cast<mpi::Rank>(1 - me);

    FMPI_CHECK(mpi::sendrecv(
        begin + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        out + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        ctx));

    chunks.emplace_back(std::make_pair(
        out + other * blocksize, out + (other + 1) * blocksize));

    op(chunks, buffer.begin());
    std::move(buffer.begin(), buffer.end(), out);
    return;
  }

  auto trace = rtlx::TimeTrace{me, "Bruck_interleave"};

  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  trace.tick(detail::ROTATE);

  // O(p * blocksize)
  std::rotate_copy(
      begin,
      // n_first
      begin + me * blocksize,
      // last
      begin + blocksize * nr,
      // out
      out);

  trace.tock(detail::ROTATE);

  // Phase 2: Communication Rounds

  trace.tick(COMMUNICATION);

  simple_vector tmpbuf{nels + nels / 2};

  auto* sendbuf  = &tmpbuf[0];
  auto* recvbuf  = &tmpbuf[nels / 2];
  auto* mergebuf = &tmpbuf[nels];

  // We never copy more than (nr/2) blocks
  std::vector<std::size_t> blocks;
  blocks.reserve(nr / 2);

  chunks.reserve(nr);
  chunks.emplace_back(std::make_pair(out, out + blocksize));

  auto const       niter = tlx::integer_log2_ceil(nr);
  constexpr size_t one   = 1;
  for (auto&& r : range(niter)) {
    auto const j = static_cast<mpi::Rank>(one << r);

    FMPI_DBG(r);

    mpi::Rank recvfrom, sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    // a) pack blocks into a contigous send buffer
    trace.tick(detail::PACK);

    auto rng = range<std::size_t>(one, nr);

    // We exchange all blocks where the j-th bit is set
    std::copy_if(
        std::begin(rng),
        std::end(rng),
        std::back_inserter(blocks),
        [j](auto idx) { return idx & j; });

    for (std::size_t b = 0; b < blocks.size(); ++b) {
      auto const block = blocks[b];
      std::copy(
          // begin
          out + block * blocksize,
          // end
          out + block * blocksize + blocksize,
          // tmp buf
          sendbuf + b * blocksize);
    }

    trace.tock(detail::PACK);

    FMPI_DBG("send_buffer");
    FMPI_DBG_RANGE(sendbuf, sendbuf + blocksize * blocks.size());

    FMPI_CHECK(mpi::irecv(
        recvbuf,
        blocksize * blocks.size(),
        recvfrom,
        EXCH_TAG_BRUCK,
        ctx,
        &reqs[1]));

    FMPI_CHECK(mpi::isend(
        sendbuf,
        blocksize * blocks.size(),
        sendto,
        EXCH_TAG_BRUCK,
        ctx,
        &reqs[0]));

    if (r > 0) {
      trace.tick(MERGE);
      // merge chunks of last iteration...
      auto const op_first = (r == 1) ? 0 : one << (r - 1);
      FMPI_DBG(op_first);
      op(chunks, std::next(buffer.begin(), op_first));
      chunks.clear();
      trace.tock(MERGE);
    }

    FMPI_CHECK(mpi::waitall(reqs));

    FMPI_DBG("recv_buffer");
    FMPI_DBG_RANGE(recvbuf, recvbuf + blocksize * blocks.size());

    {
      auto rng     = range<std::size_t>(r);
      auto sumPow2 = std::accumulate(
          std::begin(rng),
          std::end(rng),
          1,  // init with 1
          [](auto const cur, auto const v) { return cur + (1 << v); });

      auto const nmerges = std::min(nels - sumPow2, (one << r));
      FMPI_DBG(nmerges);

      for (auto&& b : range<std::size_t>(nmerges)) {
        auto f = b * blocksize;
        auto l = (b + 1) * blocksize;
        chunks.emplace_back(std::make_pair(recvbuf + f, recvbuf + l));
      }

      FMPI_DBG(chunks);
    }

    FMPI_DBG("merge_buffer");
    FMPI_DBG_RANGE(buffer.begin(), buffer.end());

    {
      // c) unpack blocks which will be forwarded to other processors
      trace.tick(detail::UNPACK);

      for (auto&& b : range(one << r, std::max(one << r, blocks.size()))) {
        std::copy(
            recvbuf + b * blocksize,
            recvbuf + b * blocksize + blocksize,
            out + blocks[b] * blocksize);
      }

      FMPI_DBG("out_buffer");
      FMPI_DBG_RANGE(out, out + nels);

      trace.tock(detail::UNPACK);
    }

    std::swap(recvbuf, mergebuf);

    blocks.clear();
  }

  trace.tick(MERGE);

  auto const nchunks = niter;

  if (nchunks > 1) {
    //chunks.resize(nchunks);

    auto mid = buffer.begin() + 2 * blocksize;

    // the first (already merged) two chunks
    chunks.emplace_back(std::make_pair(buffer.begin(), mid));

    if (nchunks > 2) {
      // the second (already merged) two chunks
      auto last = buffer.begin() + 4 * blocksize;
      chunks.emplace_back(std::make_pair(mid, last));
    }

    auto last_chunk = std::max(2, std::int32_t(nchunks) - 1);

    RTLX_ASSERT(2 <= last_chunk);

    for (auto&& r : range<std::size_t>(2, last_chunk)) {
      auto f    = (one << r) * blocksize;
      auto l    = std::min(nels, f + blocksize);
      chunks[r] = std::make_pair(buffer.begin() + f, buffer.begin() + l);
    }

    op(chunks, out);
  }

  trace.tock(MERGE);
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_mod(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto me = ctx.rank();
  auto nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::TimeTrace{me, "Bruck_Mod"};

  trace.tick(detail::ROTATE);

  auto nels = size_t(nr) * blocksize;

  std::unique_ptr<value_t[]> tmpbuf;

  {
    if (isPow2(nr)) {
      // Phase 1: Local Rotate, out[(me + block) % nr] = begin[(me - block) %
      // nr] This procedure can be achieved efficiently in two substeps

      // a) reverse_copy all blocks
      detail::reverse_copy_strided(
          begin, begin + nr * blocksize, blocksize, out);

      // b) rotate by (n - 2 * me - 1) % n
      auto shift = mod(nr - 2 * me - 1, nr);
      std::rotate(out, out + shift * blocksize, out + nels);
    }
    else {
      for (auto&& block : range<int>(nr)) {
        auto dst = mod<int>(me + block, nr);
        auto src = mod<int>(me - block, nr);

        std::copy(
            begin + src * blocksize,
            begin + (src + 1) * blocksize,
            out + dst * blocksize);
      }
    }

    // Phase 2: Communication Rounds
    tmpbuf.reset(new value_t[nels]);
  }
  trace.tock(detail::ROTATE);

  auto sendbuf = &tmpbuf[0];
  auto recvbuf = &tmpbuf[nels / 2];

  trace.tick(COMMUNICATION);

  // range = [0..log2(nr)]
  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom, sendto;

    FMPI_DBG(r);

    // In contrast to classic Bruck, sender and receiver are swapped
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me + j, static_cast<mpi::Rank>(nr)),
        mod(me - j, static_cast<mpi::Rank>(nr)));

    FMPI_DBG(sendto);
    FMPI_DBG(recvfrom);

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

    trace.tick(detail::PACK);

    {
      for (std::size_t block = me; block < me + nr; ++block) {
        //
        auto myblock = block - me;
        auto myidx   = block % nr;
        if (myblock & j) {
          FMPI_DBG(myblock);
          FMPI_DBG(myidx);
          std::copy(
              // begin
              out + myidx * blocksize,
              // end
              out + myidx * blocksize + blocksize,
              // tmp buf
              sendbuf + count * blocksize);
          ++count;
        }
      }
    }

    trace.tock(detail::PACK);

    // b) exchange
    FMPI_CHECK(mpi::sendrecv(
        sendbuf,
        blocksize * count,
        sendto,
        EXCH_TAG_BRUCK,
        recvbuf,
        blocksize * count,
        recvfrom,
        EXCH_TAG_BRUCK,
        ctx));

    // c) unpack blocks into recv buffer
    count = 0;

    trace.tick(detail::UNPACK);

    {
      for (std::size_t block = recvfrom; block < recvfrom + nr; ++block) {
        // Map from block to their idx
        auto theirblock = block - recvfrom;
        if (theirblock & j) {
          auto myblock = (theirblock + me) % nr;
          FMPI_DBG(theirblock);
          FMPI_DBG(myblock);
          std::copy(
              recvbuf + count * blocksize,
              recvbuf + count * blocksize + blocksize,
              out + myblock * blocksize);
          ++count;
        }
      }
    }
    trace.tock(detail::UNPACK);
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto nb = range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(nb),
      std::end(nb),
      std::back_inserter(chunks),
      [buf = out, blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, tmpbuf.get());

  // switch buffer back to output iterator
  std::move(tmpbuf.get(), tmpbuf.get() + nels, out);

  trace.tock(MERGE);
}
}  // namespace fmpi

#endif
