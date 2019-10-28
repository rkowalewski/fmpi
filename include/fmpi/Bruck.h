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

namespace fmpi {

template <class InputIt, class OutputIt, class Op>
inline void bruck(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  using rank_t  = int;
  auto const me = ctx.rank();
  auto const nr = static_cast<mpi::mpi_rank>(ctx.size());

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::TimeTrace{me, "Bruck"};
  trace.tick(COMMUNICATION);

  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  // O(p * blocksize)
  std::rotate_copy(
      begin,
      // n_first
      begin + me * blocksize,
      // last
      begin + blocksize * nr,
      // out
      out);

  // Phase 2: Communication Rounds

  // Reverse a buffer for send-recv exchanges
  // We never exchange more than (N/2) elements per round, so this buffer
  // suffices
  auto                       nels = size_t(nr) * blocksize;
  std::unique_ptr<value_t[]> tmpbuf{new value_t[nels]};

  auto* sendbuf = &tmpbuf[0];
  auto* recvbuf = &tmpbuf[nels / 2];

  auto n_rounds = std::ceil(std::log2(nr));
  for (std::size_t idx = 0; idx < n_rounds; ++idx) {
    auto      j = (1 << idx);
    mpi::Rank recvfrom, sendto;

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        static_cast<mpi::Rank>(mod(me - j, nr)),
        static_cast<mpi::Rank>(mod(me + j, nr)));

    // We exchange all blocks where the j-th bit is set

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

    for (rank_t block = 1; block < nr; ++block) {
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

    FMPI_CHECK(mpi::sendrecv(
        sendbuf,
        blocksize * count,
        sendto,
        100,
        recvbuf,
        blocksize * count,
        recvfrom,
        100,
        ctx));

    // c) unpack blocks into recv buffer
    count = 0;
    for (rank_t block = 1; block < nr; ++block) {
      if (block & j) {
        std::copy(
            recvbuf + count * blocksize,
            recvbuf + count * blocksize + blocksize,
            out + block * blocksize);
        ++count;
      }
    }
  }

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

  // Reverse these blocks
  for (rank_t block = 0; block < nr; ++block) {
    std::copy(
        sendbuf + (nr - block - 1) * blocksize,
        sendbuf + (nr - block) * blocksize,
        out + block * blocksize);
  }
  trace.tock(COMMUNICATION);
  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = tmpbuf.get(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, out);

  trace.tock(MERGE);

  RTLX_ASSERT(std::is_sorted(out, out + nr * blocksize));
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_mod(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  using rank_t = int;
  auto me      = ctx.rank();
  auto nr      = static_cast<rank_t>(ctx.size());

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::TimeTrace{me, "Bruck_Mod"};
  trace.tick(COMMUNICATION);

  auto nels = size_t(nr) * blocksize;

  std::unique_ptr<value_t[]> tmpbuf;

  {
    // Phase 1: Local Rotate, out[(me + block) % nr] = begin[(me - block) %
    // nr] This procedure can be achieved efficiently in two substeps

    // a) reverse_copy all blocks
    for (rank_t block = 0; block < nr; ++block) {
      std::copy(
          begin + (nr - block - 1) * blocksize,
          begin + (nr - block) * blocksize,
          out + block * blocksize);
    }

    // b) rotate by (n - 2 * me - 1) % n
    auto shift = mod(nr - 2 * me - 1, nr);
    std::rotate(out, out + shift * blocksize, out + nels);

    // Phase 2: Communication Rounds
    tmpbuf.reset(new value_t[nels]);
  }

  auto sendbuf = &tmpbuf[0];
  auto recvbuf = &tmpbuf[nels / 2];

  // range = [0..log2(nr)]
  for (auto r = 0; r < std::ceil(std::log2(nr)); ++r) {
    auto      j = (1 << r);
    mpi::Rank recvfrom, sendto;

    // In contrast to classic Bruck, sender and receiver are swapped
    std::tie(recvfrom, sendto) = std::make_pair(
        static_cast<mpi::Rank>(mod(me + j, nr)),
        static_cast<mpi::Rank>(mod(me - j, nr)));

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

    {
      for (auto block = me; static_cast<int>(block) < me + nr; ++block) {
        //
        auto myblock = block - me;
        auto myidx   = block % nr;
        if (myblock & j) {
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

    // b) exchange
    FMPI_CHECK(mpi::sendrecv(
        sendbuf,
        blocksize * count,
        sendto,
        100,
        recvbuf,
        blocksize * count,
        recvfrom,
        100,
        ctx));

    // c) unpack blocks into recv buffer
    count = 0;

    {
      for (int block = recvfrom; block < recvfrom + nr; ++block) {
        // Map from block to their idx
        auto theirblock = block - recvfrom;
        if (theirblock & j) {
          auto myblock = (theirblock + me) % nr;
          std::copy(
              recvbuf + count * blocksize,
              recvbuf + count * blocksize + blocksize,
              out + myblock * blocksize);
          ++count;
        }
      }
    }
  }
  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  std::vector<std::pair<InputIt, InputIt>> chunks;
  chunks.reserve(nr);

  auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

  std::transform(
      std::begin(range),
      std::end(range),
      std::back_inserter(chunks),
      [buf = tmpbuf.get(), blocksize](auto offset) {
        auto f = std::next(buf, offset);
        auto l = std::next(f, blocksize);
        return std::make_pair(f, l);
      });

  op(chunks, out);

  trace.tock(MERGE);

  RTLX_ASSERT(std::is_sorted(out, out + nr * blocksize));
}
}  // namespace fmpi

#endif
