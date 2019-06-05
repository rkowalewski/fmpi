#ifndef BRUCK_H
#define BRUCK_H

#include <mpi.h>

#include <cmath>
#include <memory>

#include <Debug.h>
#include <Math.h>
#include <Types.h>

namespace alltoall {

template <class InputIt, class OutputIt, class Op>
inline void bruck(InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&&)
{
  using rank_t = int;
  rank_t me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_t = typename std::iterator_traits<InputIt>::value_type;

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
  std::unique_ptr<value_t[]> send_recv_buf{new value_t[nels]};

  auto* send_buf = &send_recv_buf[0];
  auto* recv_buf = &send_recv_buf[nels / 2];

  auto n_rounds = std::ceil(std::log2(nr));
  for (std::size_t idx = 0; idx < n_rounds; ++idx) {
    auto j = (1 << idx);
    int  src, dst;

    // We send to (r + j)
    std::tie(src, dst) = std::make_pair(mod(me - j, nr), mod(me + j, nr));

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
            send_buf + count * blocksize);
        ++count;
      }
    }

    // b) exchange
    auto res = MPI_Sendrecv(
        send_buf,
        blocksize * count,
        mpi::mpi_datatype<value_t>::type(),
        dst,
        100,
        recv_buf,
        blocksize * count,
        mpi::mpi_datatype<value_t>::type(),
        src,
        100,
        comm,
        MPI_STATUS_IGNORE);
    ASSERT(res == MPI_SUCCESS);

    // c) unpack blocks into recv buffer
    count = 0;
    for (rank_t block = 1; block < nr; ++block) {
      if (block & j) {
        std::copy(
            recv_buf + count * blocksize,
            recv_buf + count * blocksize + blocksize,
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
      send_buf);

  // Reverse these blocks
  for (rank_t block = 0; block < nr; ++block) {
    std::copy(
        send_buf + (nr - block - 1) * blocksize,
        send_buf + (nr - block) * blocksize,
        out + block * blocksize);
  }
}

template <class InputIt, class OutputIt, class Op>
inline void bruck_mod(
    InputIt begin, OutputIt out, int blocksize, MPI_Comm comm, Op&&)
{
  using rank_t = int;
  rank_t me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  using value_t = typename std::iterator_traits<InputIt>::value_type;

  auto nels = size_t(nr) * blocksize;

  std::unique_ptr<value_t[]> send_recv_buf;

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
    send_recv_buf.reset(new value_t[nels]);
  }

  auto send_buf = &send_recv_buf[0];
  auto recv_buf = &send_recv_buf[nels / 2];

  // range = [0..log2(nr)]
  for (auto r = 0; r < std::ceil(std::log2(nr)); ++r) {
    auto j = (1 << r);
    int  src, dst;

    // In contrast to classic Bruck, sender and receiver are swapped
    std::tie(src, dst) = std::make_pair(mod(me + j, nr), mod(me - j, nr));

    // a) pack blocks into a contigous send buffer
    size_t count = 0;

    {
      for (auto block = me; block < me + nr; ++block) {
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
              send_buf + count * blocksize);
          ++count;
        }
      }
    }

    // b) exchange
    auto res = MPI_Sendrecv(
        send_buf,
        blocksize * count,
        mpi::mpi_datatype<value_t>::type(),
        dst,
        100,
        recv_buf,
        blocksize * count,
        mpi::mpi_datatype<value_t>::type(),
        src,
        100,
        comm,
        MPI_STATUS_IGNORE);
    ASSERT(res == MPI_SUCCESS);

    // c) unpack blocks into recv buffer
    count = 0;

    {
      for (int block = src; block < src + nr; ++block) {
        // Map from block to their idx
        auto theirblock = block - src;
        if (theirblock & j) {
          auto myblock = (theirblock + me) % nr;
          std::copy(
              recv_buf + count * blocksize,
              recv_buf + count * blocksize + blocksize,
              out + myblock * blocksize);
          ++count;
        }
      }
    }
  }
}
}  // namespace alltoall

#endif
