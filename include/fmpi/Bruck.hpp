#ifndef FMPI_BRUCK_HPP
#define FMPI_BRUCK_HPP

#include <mpi.h>

#include <cmath>
#include <fmpi/Config.hpp>
#include <fmpi/Math.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/Request.hpp>
#include <memory>
#include <numeric>
#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>
#include <tlx/math/integer_log2.hpp>
#include <tlx/simple_vector.hpp>

#include <fmpi/alltoall/Detail.hpp>

namespace fmpi {

namespace detail {

template <class BidirIt, class OutputIt>
#ifndef _OPENMP
constexpr
#endif
    OutputIt
    reverse_copy_strided(
        BidirIt     first,
        BidirIt     last,
        std::size_t blocksize,
        OutputIt    d_first) {
  auto const n = std::distance(first, last);
  RTLX_ASSERT(n % blocksize == 0);

  auto const nb = n / blocksize;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t block = 0; block < nb; ++block) {
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

template <class T>
using buffer_t = tlx::SimpleVector<T, tlx::SimpleVectorMode::Normal>;


}  // namespace detail

template <class InputIt, class OutputIt, class Op>
inline void bruck(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::Trace{"Bruck"};

  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  auto const nels = size_t(nr) * blocksize;

  std::vector<std::size_t>                     blocks;
  detail::buffer_t<value_t>                    tmpbuf{};
  typename detail::buffer_t<value_t>::iterator sendbuf{};
  typename detail::buffer_t<value_t>::iterator recvbuf{};

  {
    rtlx::TimeTrace tt{trace, detail::ROTATE};

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
    tmpbuf = detail::buffer_t<value_t>{nels};

    sendbuf = &tmpbuf[0];
    recvbuf = &tmpbuf[nels / 2];

    blocks.reserve(nr / 2);
  }

  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom;
    mpi::Rank sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    {
      rtlx::TimeTrace tt{trace, detail::PACK};

      // We exchange all blocks where the j-th bit is set
      auto rng = range<std::size_t>(1, nr);

      std::copy_if(
          std::begin(rng),
          std::end(rng),
          std::back_inserter(blocks),
          [j](auto idx) { return idx & j; });

      // a) pack blocks into a contigous send buffer

#pragma omp parallel for
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
    }

    {
      rtlx::TimeTrace tt{trace, COMMUNICATION};

      FMPI_CHECK_MPI(mpi::isend(
          sendbuf,
          blocksize * blocks.size(),
          sendto,
          EXCH_TAG_BRUCK,
          ctx,
          &reqs[0]));

      FMPI_CHECK_MPI(mpi::irecv(
          recvbuf,
          blocksize * blocks.size(),
          recvfrom,
          EXCH_TAG_BRUCK,
          ctx,
          &reqs[1]));

      FMPI_CHECK_MPI(mpi::waitall(reqs.begin(), reqs.end()));
    }

    {
      rtlx::TimeTrace tt{trace, detail::UNPACK};

      // c) unpack blocks into recv buffer
#pragma omp parallel for
      for (std::size_t b = 0; b < blocks.size(); ++b) {
        auto const block = blocks[b];
        std::copy(
            recvbuf + b * blocksize,
            recvbuf + b * blocksize + blocksize,
            out + block * blocksize);
      }

      blocks.clear();
    }
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

  {
    rtlx::TimeTrace tt{trace, COMPUTATION};

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
  }
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_indexed(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::Trace{"Bruck_indexed"};

  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  auto const nels = size_t(nr) * blocksize;

  // Reverse buffers for send-recv exchanges
  // We never exchange more than (N/2) elements per round, so this buffer
  // suffices
  std::vector<std::size_t> blocks;
  blocks.reserve(nr / 2);

  std::vector<int>          displs(nr / 2);
  detail::buffer_t<value_t> tmpbuf{nels};

  MPI_Datatype packed{};
  MPI_Count    mysize{};

  {
    rtlx::TimeTrace tt{trace, detail::ROTATE};

    // O(p * blocksize)
    std::rotate_copy(
        begin,
        // n_first
        begin + me * blocksize,
        // last
        begin + blocksize * nr,
        // out
        out);
  }

  // Phase 2: Communication Rounds
  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(1 << r);
    mpi::Rank recvfrom;

    mpi::Rank sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    {
      rtlx::TimeTrace tt{trace, detail::PACK};

      // We exchange all blocks where the j-th bit is set
      auto rng = range<std::size_t>(1, nr);

      std::copy_if(
          std::begin(rng),
          std::end(rng),
          std::back_inserter(blocks),
          [j](auto idx) { return idx & j; });

      // a) pack blocks into a contigous send buffer
      FMPI_DBG(blocks.size());

#pragma omp parallel for
      for (std::size_t b = 0; b < blocks.size(); ++b) {
        auto const block = blocks[b];
        displs[b]        = block * blocksize;

        // We can also use MPI, see below but it seems to be quite slow
        std::copy(
            // begin
            out + block * blocksize,
            // end
            out + block * blocksize + blocksize,
            // tmp buf
            tmpbuf.begin() + b * blocksize);
      }
#if 0

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
#endif

      FMPI_CHECK_MPI(MPI_Type_create_indexed_block(
          blocks.size(),
          blocksize,
          displs.data(),
          mpi::type_mapper<value_t>::type(),
          &packed));

      FMPI_CHECK_MPI(MPI_Type_commit(&packed));

      FMPI_CHECK_MPI(MPI_Type_size_x(packed, &mysize));

      RTLX_ASSERT(
          static_cast<size_t>(mysize) ==
          blocks.size() * blocksize * sizeof(value_t));
    }

    {
      rtlx::TimeTrace tt{trace, COMMUNICATION};

      FMPI_CHECK_MPI(mpi::irecv(
          out, 1, packed, recvfrom, EXCH_TAG_BRUCK, ctx.mpiComm(), &reqs[0]));

      FMPI_CHECK_MPI(mpi::isend(
          tmpbuf.begin(),
          mysize,
          MPI_BYTE,
          sendto,
          EXCH_TAG_BRUCK,
          ctx.mpiComm(),
          &reqs[1]));

      FMPI_CHECK_MPI(mpi::waitall(reqs.begin(), reqs.end()));
      FMPI_CHECK_MPI(MPI_Type_free(&packed));
    }

    blocks.clear();
  }

  {
    rtlx::TimeTrace tt{trace, COMPUTATION};

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
  }
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_interleave(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  auto const        me = ctx.rank();
  std::size_t const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  std::vector<std::pair<InputIt, InputIt>> chunks;

  auto const                nels = nr * blocksize;
  detail::buffer_t<value_t> buffer{nels};

  auto trace = rtlx::Trace{"Bruck_interleave"};

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }


  // Phase 1: Process i rotates local elements by i blocks to the left in a
  // cyclic manner.

  {
    rtlx::TimeTrace tt{trace, detail::ROTATE};

    // O(p * blocksize)
    std::rotate_copy(
        begin,
        // n_first
        begin + me * blocksize,
        // last
        begin + blocksize * nr,
        // out
        out);
  }

  // Phase 2: Communication Rounds

  detail::buffer_t<value_t> tmpbuf{nels + nels / 2};

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

  for (auto&& r : range(niter)) {
    auto const j = static_cast<mpi::Rank>(one << r);

    FMPI_DBG(r);

    mpi::Rank recvfrom;

    mpi::Rank sendto;

    auto reqs =
        std::array<MPI_Request, 2>{MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    // We send to (r + j)
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me - j, static_cast<mpi::Rank>(nr)),
        mod(me + j, static_cast<mpi::Rank>(nr)));

    // a) pack blocks into a contigous send buffer
    {
      rtlx::TimeTrace tt{trace, detail::PACK};

      auto rng = range<std::size_t>(one, nr);

      // We exchange all blocks where the j-th bit is set
      std::copy_if(
          std::begin(rng),
          std::end(rng),
          std::back_inserter(blocks),
          [j](auto idx) { return idx & j; });

#pragma omp parallel for
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
    }

    FMPI_DBG("send_buffer");
    FMPI_DBG_RANGE(sendbuf, sendbuf + blocksize * blocks.size());

    {
      rtlx::TimeTrace tt{trace, COMMUNICATION};

      FMPI_CHECK_MPI(mpi::irecv(
          recvbuf,
          blocksize * blocks.size(),
          recvfrom,
          EXCH_TAG_BRUCK,
          ctx,
          &reqs[1]));

      FMPI_CHECK_MPI(mpi::isend(
          sendbuf,
          blocksize * blocks.size(),
          sendto,
          EXCH_TAG_BRUCK,
          ctx,
          &reqs[0]));
    }

    if (r > 0) {
      {
        rtlx::TimeTrace tt{trace, COMPUTATION};
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
      }
    }

    {
      rtlx::TimeTrace tt{trace, COMMUNICATION};
      FMPI_CHECK_MPI(mpi::waitall(reqs.begin(), reqs.end()));

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
    }

    {
      // c) unpack blocks which will be forwarded to other processors
      {
        rtlx::TimeTrace tt{trace, detail::UNPACK};

#pragma omp parallel for
        for (std::size_t block = one << r;
             block < std::max(one << r, blocks.size());
             ++block) {
          FMPI_DBG(block);
          std::copy(
              recvbuf + block * blocksize,
              recvbuf + block * blocksize + blocksize,
              out + blocks[block] * blocksize);
        }

        FMPI_DBG("out_buffer");
        FMPI_DBG_RANGE(out, out + nels);

        blocks.clear();
      }
    }

    std::swap(recvbuf, mergebuf);
  }

  {
    rtlx::TimeTrace tt{trace, COMPUTATION};

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
  }
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_mod(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  auto const me = ctx.rank();
  auto const nr = ctx.size();

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto trace = rtlx::Trace{"Bruck_Mod"};

  auto const nels = size_t(nr) * blocksize;

  {
    rtlx::TimeTrace tt{trace, detail::ROTATE};

    // TODO(rkowalewski): this can be more efficient
    if (isPow2(nr)) {
      // Phase 1: Local Rotate, out[(me + block) % nr] = begin[(me - block)
      // % nr] This procedure can be achieved efficiently in two substeps

      // a) reverse_copy all blocks
      detail::reverse_copy_strided(
          begin, begin + nr * blocksize, blocksize, out);

      // b) rotate by (n - 2 * me - 1) % n
      auto shift = mod(nr - 2 * me - 1, nr);
      std::rotate(out, out + shift * blocksize, out + nels);
    }
#if 0
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
#endif
  }

  detail::buffer_t<value_t> tmpbuf{nels};
  auto*                     sendbuf = &tmpbuf[0];
  auto*                     recvbuf = &tmpbuf[nels / 2];

  constexpr std::size_t one = 1;

  std::vector<std::size_t> blocks;
  blocks.reserve(nr / 2);

  // range = [0..log2(nr)]
  for (auto&& r : range(tlx::integer_log2_ceil(nr))) {
    auto      j = static_cast<mpi::Rank>(one << r);
    mpi::Rank recvfrom;
    mpi::Rank sendto;

    FMPI_DBG(r);

    // In contrast to classic Bruck, sender and receiver are swapped
    std::tie(recvfrom, sendto) = std::make_pair(
        mod(me + j, static_cast<mpi::Rank>(nr)),
        mod(me - j, static_cast<mpi::Rank>(nr)));

    FMPI_DBG(sendto);
    FMPI_DBG(recvfrom);

    // a) pack blocks into a contigous send buffer

    {
      rtlx::TimeTrace tt{trace, detail::PACK};

      // We exchange all blocks where the j-th bit is set
      for (auto&& idx : range<std::size_t>(me + (me == 0), me + nr)) {
        if ((idx - me) & static_cast<std::size_t>(j)) {
          blocks.emplace_back(idx % nr);
        }
      }

      FMPI_DBG(blocks);

#pragma omp parallel for
      for (std::size_t i = 0; i < blocks.size(); ++i) {
        auto myidx = blocks[i];
        std::copy(
            // begin
            out + myidx * blocksize,
            // end
            out + myidx * blocksize + blocksize,
            // tmp buf
            sendbuf + i * blocksize);
      }
    }

    {
      rtlx::TimeTrace tt{trace, COMMUNICATION};

      // b) exchange
      FMPI_CHECK_MPI(mpi::sendrecv(
          sendbuf,
          blocksize * blocks.size(),
          sendto,
          EXCH_TAG_BRUCK,
          recvbuf,
          blocksize * blocks.size(),
          recvfrom,
          EXCH_TAG_BRUCK,
          ctx));
    }

    // c) unpack blocks into recv buffer

    {
      rtlx::TimeTrace tt{trace, detail::UNPACK};
      auto            rng = range<std::size_t>(1, nr);
      // We exchange all blocks where the j-th bit is set
      std::copy_if(
          std::begin(rng), std::end(rng), std::begin(blocks), [j](auto idx) {
            return idx & static_cast<std::size_t>(j);
          });

#pragma omp parallel for
      for (std::size_t i = 0; i < blocks.size(); ++i) {
        auto const myblock = (blocks[i] + me) % nr;

        FMPI_DBG(myblock);

        std::copy(
            recvbuf + i * blocksize,
            recvbuf + i * blocksize + blocksize,
            out + myblock * blocksize);
      }
      blocks.clear();
    }
  }

  {
    rtlx::TimeTrace tt{trace, COMPUTATION};

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

    op(chunks, tmpbuf.begin());

    // switch buffer back to output iterator
    std::move(tmpbuf.begin(), tmpbuf.begin() + nels, out);
  }
}
}  // namespace fmpi

#endif
