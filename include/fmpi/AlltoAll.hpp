#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <mpi.h>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <tlx/simple_vector.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>

// Other AllToAll Algorithms

namespace fmpi {

/// Forward Declarations

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
void scatteredPairwiseWaitsome(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
inline void scatteredPairwiseWaitsomeOverlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
inline void scatteredPairwiseWaitall(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto nr = ctx.size();

  auto trace = rtlx::Trace{"AlltoAll"};

  std::unique_ptr<value_type[]> rbuf;

  {
    rtlx::TimeTrace tt(trace, COMMUNICATION);

    rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

    FMPI_CHECK_MPI(mpi::alltoall(
        std::addressof(*begin), blocksize, &rbuf[0], blocksize, ctx));
  }

  {
    rtlx::TimeTrace tt(trace, COMPUTATION);

    std::vector<std::pair<InputIt, InputIt>> chunks;
    chunks.reserve(nr);

    auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

    std::transform(
        std::begin(range),
        std::end(range),
        std::back_inserter(chunks),
        [buf = rbuf.get(), blocksize](auto offset) {
          auto f = std::next(buf, offset);
          auto l = std::next(f, blocksize);
          return std::make_pair(f, l);
        });

    op(chunks, out);
  }
}

namespace detail {

template <class InputIt, class OutputIt, class Op>
inline void scatteredPairwise_lt3(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op,
    rtlx::Trace&        trace) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;

  using merge_buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto chunks = std::vector<std::pair<InputIt, InputIt>>{};
  chunks.reserve(2);

  auto const me = ctx.rank();

  chunks.emplace_back(
      std::make_pair(begin + me * blocksize, begin + (me + 1) * blocksize));

  if (ctx.size() == 1) {
    rtlx::TimeTrace tt(trace, COMPUTATION);
    op(chunks, out);
    return;
  }

  auto other = static_cast<mpi::Rank>(1 - me);

  {
    rtlx::TimeTrace tt(trace, COMMUNICATION);
    FMPI_CHECK_MPI(mpi::sendrecv(
        begin + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        out + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        ctx));
  }

  {
    rtlx::TimeTrace tt(trace, COMPUTATION);

    chunks.emplace_back(std::make_pair(
        out + other * blocksize, out + (other + 1) * blocksize));
    merge_buffer_t buffer{ctx.size() * blocksize};
    op(chunks, buffer.begin());

    std::move(buffer.begin(), buffer.end(), out);
  }

  trace.put(N_COMM_ROUNDS, 1);
}
}  // namespace detail

}  // namespace fmpi

#include <fmpi/alltoall/Waitall.hpp>
#include <fmpi/alltoall/Waitsome.hpp>

#endif
