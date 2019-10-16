#ifndef MPI__P2P_H
#define MPI__P2P_H

#include <fmpi/mpi/Types.h>

namespace mpi {

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
  auto mpi_datatype = mpi::type_mapper<S>::type();

  RTLX_ASSERT_RETURNS(
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
}  // namespace mpi

#endif
