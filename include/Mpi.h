#ifndef MPI_H__INCLUDED
#define MPI_H__INCLUDED

#include <mpi.h>
#include <array>
#include <type_traits>

#include <Types.h>

namespace a2a {
constexpr size_t REQ_SEND = 0;
constexpr size_t REQ_RECV = 1;
template <class S, class R>
inline auto sendrecv(
    const S* sbuf,
    size_t   scount,
    int      sto,
    int      stag,
    R*       rbuf,
    size_t   rcount,
    int      rfrom,
    int      rtag,
    MPI_Comm comm)
{
  std::array<MPI_Request, 2> reqs = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  static_assert(std::is_same<S, R>::value, "");

  auto mpi_datatype = mpi::mpi_datatype<S>::type();

#if 0

  // Overlapping first round...
  A2A_ASSERT_RETURNS(
      MPI_Irecv(
          rbuf,
          static_cast<int>(rcount),
          mpi_datatype,
          rfrom,
          rtag,
          comm,
          &(reqs[REQ_RECV])),
      MPI_SUCCESS);

  // Overlapping first round...
  A2A_ASSERT_RETURNS(
      MPI_Isend(
          sbuf,
          static_cast<int>(scount),
          mpi_datatype,
          sto,
          stag,
          comm,
          &(reqs[REQ_SEND])),
      MPI_SUCCESS);

#else

  A2A_ASSERT_RETURNS(
      MPI_Sendrecv(
          sbuf,
          static_cast<int>(scount),
          mpi_datatype,
          sto,
          stag,
          rbuf,
          rcount,
          mpi_datatype,
          rfrom,
          rtag,
          comm,
          MPI_STATUSES_IGNORE),
      MPI_SUCCESS);

#endif

  return reqs;
}

}  // namespace a2a

#endif
