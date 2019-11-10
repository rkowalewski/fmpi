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

  // Reverse a buffer for send-recv exchanges
  // We never exchange more than (N/2) elements per round, so this buffer
  // suffices
  auto                       nels = size_t(nr) * blocksize;
  std::unique_ptr<value_t[]> tmpbuf{new value_t[nels]};

  auto* sendbuf = &tmpbuf[0];
  auto* recvbuf = &tmpbuf[nels / 2];

  std::vector<std::size_t> blocks;
  blocks.reserve(nr / 2);

  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom, sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    trace.tick(detail::PACK);

    // We exchange all blocks where the j-th bit is set
    auto rng = range<std::size_t>(1, nr);

    // We exchange all blocks where the j-th bit is set
    std::copy_if(
        std::begin(rng),
        std::end(rng),
        std::back_inserter(blocks),
        [j](auto idx) { return idx & j; });

    // a) pack blocks into a contigous send buffer

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

    trace.tick(COMMUNICATION);

    FMPI_CHECK(mpi::isend(
        sendbuf,
        blocksize * blocks.size(),
        sendto,
        EXCH_TAG_BRUCK,
        ctx,
        &reqs[0]));

    FMPI_CHECK(mpi::irecv(
        recvbuf,
        blocksize * blocks.size(),
        recvfrom,
        EXCH_TAG_BRUCK,
        ctx,
        &reqs[1]));

    FMPI_CHECK(mpi::waitall(reqs));
    trace.tock(COMMUNICATION);

    trace.tick(detail::UNPACK);

    // c) unpack blocks into recv buffer
    for (std::size_t b = 0; b < blocks.size(); ++b) {
      auto const block = blocks[b];
      std::copy(
          recvbuf + b * blocksize,
          recvbuf + b * blocksize + blocksize,
          out + block * blocksize);
    }

    blocks.clear();
  }

  trace.tock(detail::UNPACK);

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
inline void bruck_indexed(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::TimeTrace{me, "Bruck_indexed"};

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

  // Reverse a buffer for send-recv exchanges
  // We never exchange more than (N/2) elements per round, so this buffer
  // suffices
  auto const                 nels = size_t(nr) * blocksize;
  std::unique_ptr<value_t[]> tmpbuf{new value_t[nels]};

  std::vector<int> displs(nr / 2);

  auto const type = mpi::type_mapper<value_t>::type();

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
        displs[count] = block * blocksize;
        ++count;
      }
    }

    FMPI_DBG(count);

    MPI_Datatype packed;
    MPI_Type_create_indexed_block(
        count, blocksize, displs.data(), type, &packed);
    MPI_Type_commit(&packed);

    MPI_Count mysize;
    MPI_Type_size_x(packed, &mysize);
    FMPI_DBG(mysize);

    RTLX_ASSERT(
        static_cast<size_t>(mysize) == count * blocksize * sizeof(value_t));

    RTLX_ASSERT_RETURNS(
        MPI_Sendrecv(
            out,
            1,
            packed,
            me,
            EXCH_TAG_BRUCK,
            tmpbuf.get(),
            mysize,
            MPI_BYTE,
            me,
            EXCH_TAG_BRUCK,
            ctx.mpiComm(),
            MPI_STATUS_IGNORE),
        MPI_SUCCESS);
    trace.tock(detail::PACK);

    FMPI_DBG("tmpbuf");
    FMPI_DBG_RANGE(tmpbuf.get(), tmpbuf.get() + count * blocksize);

    trace.tick(COMMUNICATION);

    FMPI_CHECK(mpi::irecv_type(
        out, 1, packed, recvfrom, EXCH_TAG_BRUCK, ctx, &reqs[0]));

    FMPI_CHECK(mpi::isend_type(
        tmpbuf.get(),
        mysize,
        MPI_BYTE,
        sendto,
        EXCH_TAG_BRUCK,
        ctx,
        &reqs[1]));

    FMPI_CHECK(mpi::waitall(reqs));
    trace.tock(COMMUNICATION);

    FMPI_DBG("out");
    FMPI_DBG_RANGE(out, out + nels);

    MPI_Type_free(&packed);
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
  auto const        me = ctx.rank();
  std::size_t const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;
  using simple_vector =
      tlx::SimpleVector<value_t, tlx::SimpleVectorMode::Normal>;

  std::vector<std::pair<InputIt, InputIt>> chunks;

  auto const    nels = nr * blocksize;
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

  std::vector<std::ptrdiff_t> merged;
  merged.reserve(nr);
  merged.push_back(0);

  // std::size_t lastCopied = 0;

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

    trace.tick(COMMUNICATION);

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

    trace.tock(COMMUNICATION);

    if (r > 0) {
      trace.tick(MERGE);

#if 0
      if (chunks.size() > 3) {
#endif
      // merge chunks of last iteration...
      // auto const op_first = (r == 1) ? 0 : (one << (r - 1)) * blocksize;
      auto const op_first = merged.back();
      FMPI_DBG(op_first);
      FMPI_DBG(chunks.size());
      op(chunks, std::next(buffer.begin(), op_first));
      merged.push_back(merged.back() + chunks.size() * blocksize);
      chunks.clear();
      FMPI_DBG("merge_buffer");
      FMPI_DBG_RANGE(buffer.begin(), buffer.end());
#if 0
      } else {
        std::size_t count = 0;
        for (auto& c : chunks) {
          auto const dist =std::distance(c.first, c.second);
          FMPI_DBG(dist);
          FMPI_DBG(lastCopied);
          auto const target = out + count;
          std::move(c.first, c.second, target);
          c.first = target;
          c.second = target + dist;
          count += dist;
        }
      lastCopied += count;
      }
#endif
      trace.tock(MERGE);
    }

    trace.tick(COMMUNICATION);
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

      auto const nmerges = std::min(nr - sumPow2, (one << r));
      FMPI_DBG(nmerges);

      for (auto&& b : range<std::size_t>(nmerges)) {
        auto f = b * blocksize;
        auto l = (b + 1) * blocksize;
        chunks.emplace_back(std::make_pair(recvbuf + f, recvbuf + l));
      }

      FMPI_DBG(chunks);
    }
    trace.tock(COMMUNICATION);

    {
      // c) unpack blocks which will be forwarded to other processors
      trace.tick(detail::UNPACK);

      for (auto&& block :
           range(one << r, std::max(one << r, blocks.size()))) {
        FMPI_DBG(block);
        std::copy(
            recvbuf + block * blocksize,
            recvbuf + block * blocksize + blocksize,
            out + blocks[block] * blocksize);
      }

      FMPI_DBG("out_buffer");
      FMPI_DBG_RANGE(out, out + nels);

      blocks.clear();
      trace.tock(detail::UNPACK);
    }

    std::swap(recvbuf, mergebuf);
  }

  trace.tick(MERGE);

  auto const nchunks = niter;

  if (nchunks > 1) {
#if 0
    auto mid = buffer.begin() + 2 * blocksize;

    // the first (already merged) two chunks
    chunks.emplace_back(std::make_pair(buffer.begin(), mid));

    if (nchunks > 2) {
      // the second (already merged) two chunks
      auto last = buffer.begin() + 4 * blocksize;
      chunks.emplace_back(std::make_pair(mid, last));
    }
#endif
    FMPI_DBG(merged);
    std::transform(
        std::begin(merged),
        std::prev(std::end(merged)),
        std::next(std::begin(merged)),
        std::back_inserter(chunks),
        [rbuf = buffer.begin()](auto first, auto next) {
          return std::make_pair(
              std::next(rbuf, first), std::next(rbuf, next));
        });

#if 0
    auto last_chunk = std::max(2, std::int32_t(nchunks) - 1);

    RTLX_ASSERT(2 <= last_chunk);

    for (auto&& r : range<std::size_t>(2, last_chunk)) {
      auto f    = (one << r) * blocksize;
      auto l    = std::min(nels, (one << (r+1)) * blocksize);
      FMPI_DBG(std::make_pair(f, l));
      chunks.emplace_back(std::make_pair(buffer.begin() + f, buffer.begin() + l));
    }
#endif

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
