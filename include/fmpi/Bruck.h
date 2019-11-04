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

  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom, sendto;

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    // We exchange all blocks where the j-th bit is set

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

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
    for (std::size_t block = 1; block < nr; ++block) {
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
  trace.tick(COMMUNICATION);

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

  auto sendbuf = &tmpbuf[0];
  auto recvbuf = &tmpbuf[nels / 2];

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
