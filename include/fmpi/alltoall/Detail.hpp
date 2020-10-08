#ifndef FMPI_ALLTOALL_DETAIL_HPP
#define FMPI_ALLTOALL_DETAIL_HPP

#include <fmpi/Config.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/Trace.hpp>
#include <rtlx/Timer.hpp>
#include <tlx/simple_vector.hpp>
#include <vector>

namespace fmpi {
namespace detail {

template <class InputIt, class OutputIt>
inline void ring_pairwise_lt3(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    MultiTrace&         multi_trace) {
  auto const me = ctx.rank();

  FMPI_ASSERT(ctx.size() < 3);

  auto other = static_cast<mpi::Rank>((ctx.size()) == 1 ? me : 1 - me);

  {
    steady_timer t{multi_trace.duration(kCommunicationTime)};
    FMPI_CHECK_MPI(mpi::sendrecv(
        begin + other * blocksize,
        blocksize,
        other,
        kTagBruck,
        out + other * blocksize,
        blocksize,
        other,
        kTagBruck,
        ctx));
  }
}
}  // namespace detail
}  // namespace fmpi
#endif
