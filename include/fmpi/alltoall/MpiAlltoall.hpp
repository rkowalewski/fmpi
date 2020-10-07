#ifndef FMPI_ALLTOALL_MPIALLTOALL_HPP
#define FMPI_ALLTOALL_MPIALLTOALL_HPP

#include <mpi.h>

#include <fmpi/Config.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/detail/Assert.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/Trace.hpp>
#include <rtlx/Timer.hpp>
#include <tlx/simple_vector.hpp>

namespace fmpi {

constexpr auto kAlltoall = std::string_view("AlltoAll");

template <class InputIt, class OutputIt, class Op>
collective_future mpi_alltoall(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto nr = ctx.size();

  auto trace = MultiTrace{kAlltoall};

  std::unique_ptr<value_type[]> rbuf;

  {
    steady_timer tt{trace.duration(kCommunicationTime)};

    rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

    auto request = std::make_unique<MPI_Request>();

    FMPI_CHECK_MPI(MPI_Ialltoall(
        std::addressof(*begin),
        static_cast<int>(blocksize),
        mpi::type_mapper<value_type>::type(),
        rbuf.get(),
        static_cast<int>(blocksize),
        mpi::type_mapper<value_type>::type(),
        ctx.mpiComm(),
        request.get()));

    auto future = make_mpi_future(std::move(request));
  }

  {
    steady_timer tt{trace.duration(kComputationTime)};

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

  return make_ready_future(MPI_SUCCESS);
}
}  // namespace fmpi
#endif
