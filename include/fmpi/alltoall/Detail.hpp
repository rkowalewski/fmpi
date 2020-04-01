#ifndef FMPI_ALLTOALL_DETAIL_HPP
#define FMPI_ALLTOALL_DETAIL_HPP

#include <fmpi/Config.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Trace.hpp>

#include <tlx/simple_vector.hpp>

namespace fmpi {
namespace detail {

template <class InputIt, class OutputIt, class Op>
inline void Ring_lt3(
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
#endif
