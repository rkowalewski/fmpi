#ifndef FMPI_DETAIL_COLLECTIVE_ARGS_HPP
#define FMPI_DETAIL_COLLECTIVE_ARGS_HPP

#include <fmpi/mpi/Environment.hpp>

namespace fmpi {
namespace detail {

struct CollectiveArgs {
  const void* const   sendbuf;
  std::size_t const   sendcount;
  MPI_Datatype const  sendtype;
  void* const         recvbuf;
  std::size_t const   recvcount;
  MPI_Datatype const  recvtype;
  mpi::Context const& comm;

  constexpr CollectiveArgs(
      const void*         sendbuf_,
      std::size_t         sendcount_,
      MPI_Datatype        sendtype_,
      void*               recvbuf_,
      std::size_t         recvcount_,
      MPI_Datatype        recvtype_,
      mpi::Context const& comm_)
    : sendbuf(sendbuf_)
    , sendcount(sendcount_)
    , sendtype(sendtype_)
    , recvbuf(recvbuf_)
    , recvcount(recvcount_)
    , recvtype(recvtype_)
    , comm(comm_) {
  }
};

}  // namespace detail
}  // namespace fmpi

#endif
