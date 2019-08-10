#ifndef MPI_H
#define MPI_H

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
  auto mpi_datatype = mpi::mpi_datatype<S>::type();

  A2A_ASSERT_RETURNS(
      MPI_Sendrecv(
          sbuf,
          static_cast<int>(scount),
          mpi_datatype,
          sto,
          stag,
          rbuf,
          static_cast<int>(rcount),
          mpi_datatype,
          rfrom,
          rtag,
          comm,
          MPI_STATUSES_IGNORE),
      MPI_SUCCESS);
}

}  // namespace a2a

#endif
