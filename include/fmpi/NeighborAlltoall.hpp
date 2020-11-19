#ifndef FMPI_NEIGHBOR_ALLTOALL_HPP
#define FMPI_NEIGHBOR_ALLTOALL_HPP

#include <fmpi/Schedule.hpp>
#include <fmpi/concurrency/Future.hpp>

namespace fmpi {
namespace detail {

class NeighborAlltoallCtx {
 public:
  NeighborAlltoallCtx(
      const void*         sendbuf_,
      std::size_t         sendcount_,
      MPI_Datatype        sendtype_,
      void*               recvbuf_,
      std::size_t         recvcount_,
      MPI_Datatype        recvtype_,
      mpi::Context const& comm_,
      ScheduleOpts const& opts_);

  collective_future execute();

 private:
  const void*         sendbuf;
  std::size_t const   sendcount;
  MPI_Datatype const  sendtype;
  void*               recvbuf;
  std::size_t const   recvcount;
  MPI_Datatype const  recvtype;
  mpi::Context const& comm;
  MPI_Aint            recvextent{};
  MPI_Aint            sendextent{};
  ScheduleOpts const& opts;
};

}  // namespace detail
}  // namespace fmpi

#endif
