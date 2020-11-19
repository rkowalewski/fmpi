#include <fmpi/Debug.hpp>
#include <fmpi/Exception.hpp>
#include <fmpi/Message.hpp>
#include <fmpi/NeighborAlltoall.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/NumericRange.hpp>

namespace fmpi {
namespace detail {

using namespace std::literals::string_view_literals;
// constexpr auto t_copy = "Tcomm.local_copy"sv;

NeighborAlltoallCtx::NeighborAlltoallCtx(
    const void*         sendbuf_,
    size_t              sendcount_,
    MPI_Datatype        sendtype_,
    void*               recvbuf_,
    std::size_t         recvcount_,
    MPI_Datatype        recvtype_,
    mpi::Context const& comm_,
    ScheduleOpts const& opts_)
  : sendbuf(sendbuf_)
  , sendcount(sendcount_)
  , sendtype(sendtype_)
  , recvbuf(recvbuf_)
  , recvcount(recvcount_)
  , recvtype(recvtype_)
  , comm(comm_)
  , opts(opts_) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(recvtype == sendtype);
  MPI_Type_get_extent(recvtype, &recvlb, &recvextent);
  MPI_Type_get_extent(sendtype, &sendlb, &sendextent);

  int topo_t;
  FMPI_CHECK_MPI(MPI_Topo_test(comm.mpiComm(), &topo_t));

  if (topo_t != MPI_CART) {
    throw fmpi::NotImplemented{};
  }
}
#if 0


    if( 0 == cart->ndims ) return OMPI_SUCCESS;

    ompi_datatype_get_extent(rdtype, &lb, &rdextent);
    ompi_datatype_get_extent(sdtype, &lb, &sdextent);
    reqs = preqs = ompi_coll_base_comm_get_reqs( module->base_data, 4 * cart->ndims);
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* post receives first */
    for (dim = 0, nreqs = 0; dim < cart->ndims ; ++dim) {
        int srank = MPI_PROC_NULL, drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }

        if (MPI_PROC_NULL != srank) {
            nreqs++;
            rc = MCA_PML_CALL(irecv(rbuf, rcount, rdtype, srank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim,
                                    comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        rbuf = (char *) rbuf + rdextent * rcount;

        if (MPI_PROC_NULL != drank) {
            nreqs++;
            rc = MCA_PML_CALL(irecv(rbuf, rcount, rdtype, drank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1,
                                    comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        rbuf = (char *) rbuf + rdextent * rcount;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs);
        return rc;
    }

    for (dim = 0 ; dim < cart->ndims ; ++dim) {
        int srank = MPI_PROC_NULL, drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }

        if (MPI_PROC_NULL != srank) {
            /* remove cast from const when the pml layer is updated to take
             * a const for the send buffer. */
            nreqs++;
            rc = MCA_PML_CALL(isend((void *) sbuf, scount, sdtype, srank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1,
                                    MCA_PML_BASE_SEND_STANDARD,
                                    comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        sbuf = (const char *) sbuf + sdextent * scount;

        if (MPI_PROC_NULL != drank) {
            nreqs++;
            rc = MCA_PML_CALL(isend((void *) sbuf, scount, sdtype, drank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim,
                                    MCA_PML_BASE_SEND_STANDARD,
                                    comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        sbuf = (const char *) sbuf + sdextent * scount;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs);
        return rc;
    }

    rc = ompi_request_wait_all (nreqs, reqs, MPI_STATUSES_IGNORE);
    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs);
    }
    return rc;
#endif

collective_future NeighborAlltoallCtx::execute() {
  int ndims;
  FMPI_CHECK_MPI(MPI_Cartdim_get(comm.mpiComm(), &ndims));

  auto dims    = fmpi::FixedVector<int>{ndims};
  auto periods = fmpi::FixedVector<int>{ndims};

  {
    auto tmp = fmpi::FixedVector<int>{ndims};
    FMPI_CHECK_MPI(MPI_Cart_get(
        comm.mpiComm(), ndims, dims.data(), periods.data(), tmp.data()));
  }

  std::size_t const neighbors = 2 * ndims;

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(neighbors);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));
  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  auto const tag_space = comm.requestTagSpace(2 * ndims);

  /* post receives first */
  for (auto&& dim : range(ndims)) {
    int srank = MPI_PROC_NULL, drank = MPI_PROC_NULL;

    if (dims[dim] > 1) {
      MPI_Cart_shift(comm.mpiComm(), dim, 1, &srank, &drank);
    } else if (1 == dims[dim] && periods[dim]) {
      srank = drank = comm.rank();
    }

    if (MPI_PROC_NULL != srank) {
      dispatcher.schedule(
          hdl,
          message_type::IRECV,
          make_receive(
              recvbuf,
              recvcount,
              recvtype,
              static_cast<mpi::Rank>(srank),
              tag_space - 2 * dim,
              comm.mpiComm()));
    }

    recvbuf = add(recvbuf, recvextent * recvcount);

    if (MPI_PROC_NULL != drank) {
      dispatcher.schedule(
          hdl,
          message_type::IRECV,
          make_receive(
              recvbuf,
              recvcount,
              recvtype,
              static_cast<mpi::Rank>(drank),
              tag_space - 2 * dim - 1,
              comm.mpiComm()));
    }

    recvbuf = add(recvbuf, recvextent * recvcount);
  }

  /* post receives first */
  for (auto&& dim : range(ndims)) {
    int srank = MPI_PROC_NULL, drank = MPI_PROC_NULL;

    if (dims[dim] > 1) {
      MPI_Cart_shift(comm.mpiComm(), dim, 1, &srank, &drank);
    } else if (1 == dims[dim] && periods[dim]) {
      srank = drank = comm.rank();
    }

    if (MPI_PROC_NULL != srank) {
      dispatcher.schedule(
          hdl,
          message_type::ISEND,
          make_send(
              sendbuf,
              sendcount,
              sendtype,
              static_cast<mpi::Rank>(srank),
              tag_space - 2 * dim - 1,
              comm.mpiComm()));
    }

    sendbuf = add(sendbuf, sendextent * sendcount);

    if (MPI_PROC_NULL != drank) {
      dispatcher.schedule(
          hdl,
          message_type::ISEND,
          make_send(
              sendbuf,
              sendcount,
              sendtype,
              static_cast<mpi::Rank>(drank),
              tag_space - 2 * dim,
              comm.mpiComm()));
    }

    sendbuf = add(sendbuf, sendextent * sendcount);
  }

  return future;
}

}  // namespace detail
}  // namespace fmpi
