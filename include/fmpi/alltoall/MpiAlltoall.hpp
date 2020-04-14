#ifndef FMPI_ALLTOALL_MPIALLTOALL_HPP
#define FMPI_ALLTOALL_MPIALLTOALL_HPP

#include <mpi.h>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>

#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <tlx/simple_vector.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>

namespace fmpi {
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
}  // namespace fmpi
#endif