#ifndef FMPI_ALLTOALL_MPIALLTOALL_HPP
#define FMPI_ALLTOALL_MPIALLTOALL_HPP


#if 0
#include <mpi.h>

#include <fmpi/Config.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/detail/Assert.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/Trace.hpp>
#include <rtlx/Timer.hpp>
#include <tlx/simple_vector.hpp>


namespace fmpi {

constexpr auto kAlltoall = std::string_view("AlltoAll");

collective_future mpi_alltoall(
    void*               sendbuf,
    std::size_t         sendcount,
    MPI_Datatype        sendtype,
    void*               recvbuf,
    std::size_t         recvcount,
    MPI_Datatype        recvtype,
    mpi::Context const& ctx,
    ScheduleOpts /*args*/) {
  auto trace = MultiTrace{kAlltoall};

  // std::unique_ptr<value_type[]> rbuf;

  //{
  steady_timer tt{trace.duration(kCommunicationTime)};

  // rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  auto request = std::make_unique<MPI_Request>();

  FMPI_CHECK_MPI(MPI_Ialltoall(
      sendbuf,
      sendcount,
      sendtype,
      recvbuf,
      recvcount,
      recvtype,
      ctx.mpiComm(),
      request.get()));

  // auto future = make_mpi_future(std::move(request));
  return make_mpi_future(std::move(request));
  //}

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
}  // namespace fmpi
}  // namespace fmpi
#endif
#endif
