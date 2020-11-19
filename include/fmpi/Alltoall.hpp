#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <fmpi/Schedule.hpp>
#include <fmpi/concurrency/Future.hpp>

namespace fmpi {
namespace detail {

class AlltoallCtx {
 public:
  AlltoallCtx(
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
  collective_future         comm_intermediate();
  [[nodiscard]] const void* send_offset(mpi::Rank r) const;
  [[nodiscard]] void*       recv_offset(mpi::Rank r) const;
  void                      local_copy();

  const void* const   sendbuf;
  std::size_t const   sendcount;
  MPI_Datatype const  sendtype;
  void* const         recvbuf;
  std::size_t const   recvcount;
  MPI_Datatype const  recvtype;
  mpi::Context const& comm;
  MPI_Aint            recvextent{};
  MPI_Aint            sendextent{};
  int32_t const       sendrecvtag{};
  ScheduleOpts const& opts;
};

}  // namespace detail

inline collective_future alltoall(
    const void*         sendbuf,
    std::size_t         sendcount,
    MPI_Datatype        sendtype,
    void*               recvbuf,
    std::size_t         recvcount,
    MPI_Datatype        recvtype,
    mpi::Context const& ctx,
    ScheduleOpts const& schedule_args) {
  auto coll = detail::AlltoallCtx{
      sendbuf,
      sendcount,
      sendtype,
      recvbuf,
      recvcount,
      recvtype,
      ctx,
      schedule_args};
  return coll.execute();
}

collective_future alltoall_tune(
    const void*         sendbuf,
    std::size_t         sendcount,
    MPI_Datatype        sendtype,
    void*               recvbuf,
    std::size_t         recvcount,
    MPI_Datatype        recvtype,
    mpi::Context const& ctx);

}  // namespace fmpi

#endif
