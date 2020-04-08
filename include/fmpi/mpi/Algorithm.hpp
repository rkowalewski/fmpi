#ifndef FMPI_MPI_ALGORITHM_HPP
#define FMPI_MPI_ALGORITHM_HPP

#include <rtlx/Assert.hpp>

#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/TypeMapper.hpp>

namespace mpi {

template <class T>
inline int irecv(
    T*             buf,
    std::size_t    count,
    Rank           source,
    int            tag,
    Context const& ctx,
    MPI_Request*   req)

{
  auto type = mpi::type_mapper<T>::type();

  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Irecv(
      buf, static_cast<int>(count), type, source, tag, ctx.mpiComm(), req);
}

template <class T>
inline int isend(
    T const*       buf,
    std::size_t    count,
    Rank           target,
    int            tag,
    Context const& ctx,
    MPI_Request*   req)

{
  auto type = mpi::type_mapper<T>::type();

  RTLX_ASSERT(count < std::numeric_limits<int>::max());

  return MPI_Isend(
      buf, static_cast<int>(count), type, target, tag, ctx.mpiComm(), req);
}

int isend(
    void const*  buf,
    std::size_t  count,
    MPI_Datatype type,
    Rank         target,
    int          tag,
    MPI_Comm     comm,
    MPI_Request* req);

int irecv(
    void*        buf,
    std::size_t  count,
    MPI_Datatype type,
    Rank         source,
    int          tag,
    MPI_Comm     comm,
    MPI_Request* req);

template <class T, class U>
inline int sendrecv(
    T const*       sendbuf,
    std::size_t    sendcount,
    Rank           dest,
    int            sendtag,
    U*             recvbuf,
    std::size_t    recvcount,
    Rank           source,
    int            recvtag,
    Context const& ctx) {
  RTLX_ASSERT(sendcount < std::numeric_limits<int>::max());
  RTLX_ASSERT(recvcount < std::numeric_limits<int>::max());

  return MPI_Sendrecv(
      sendbuf,
      static_cast<int>(sendcount),
      mpi::type_mapper<T>::type(),
      dest,
      sendtag,
      recvbuf,
      static_cast<int>(recvcount),
      mpi::type_mapper<U>::type(),
      source,
      recvtag,
      ctx.mpiComm(),
      MPI_STATUS_IGNORE);
}

template <class T, class U>
inline int alltoall(
    T const*       sendbuf,
    std::size_t    sendcount,
    U*             recvbuf,
    std::size_t    recvcount,
    Context const& ctx) {
  RTLX_ASSERT(sendcount < std::numeric_limits<int>::max());
  RTLX_ASSERT(recvcount < std::numeric_limits<int>::max());

  return MPI_Alltoall(
      sendbuf,
      static_cast<int>(sendcount),
      mpi::type_mapper<T>::type(),
      recvbuf,
      static_cast<int>(recvcount),
      mpi::type_mapper<U>::type(),
      ctx.mpiComm());
}

template <class T>
inline auto allreduce_minmax(Context const& ctx, T value) {
  auto mpi_type = mpi::type_mapper<T>::type();

  T min;

  T max;

  RTLX_ASSERT_RETURNS(
      MPI_Allreduce(&value, &min, 1, mpi_type, MPI_MIN, ctx.mpiComm()),
      MPI_SUCCESS);
  RTLX_ASSERT_RETURNS(
      MPI_Allreduce(&value, &max, 1, mpi_type, MPI_MAX, ctx.mpiComm()),
      MPI_SUCCESS);

  return std::make_pair(min, max);
}

}  // namespace mpi

#endif
