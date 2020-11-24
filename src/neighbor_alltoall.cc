#include <fmpi/Debug.hpp>
#include <fmpi/Exception.hpp>
#include <fmpi/Message.hpp>
#include <fmpi/NeighborAlltoall.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/detail/Tags.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <rtlx/ScopedLambda.hpp>

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
    mpi::Context const& comm_)
  : sendbuf(sendbuf_)
  , sendcount(sendcount_)
  , sendtype(sendtype_)
  , recvbuf(recvbuf_)
  , recvcount(recvcount_)
  , recvtype(recvtype_)
  , comm(comm_) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(recvtype == sendtype);
  MPI_Type_get_extent(recvtype, &recvlb, &recvextent);
  MPI_Type_get_extent(sendtype, &sendlb, &sendextent);

  FMPI_CHECK_MPI(MPI_Topo_test(comm.mpiComm(), &topology));
}

collective_future NeighborAlltoallCtx::alltoall_cartesian() {
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

  // setup dispatcher
  auto&      dispatcher = static_dispatcher_pool();
  auto const hdl        = dispatcher.submit(std::move(schedule_state));
  auto       finalizer =
      rtlx::scope_exit([&dispatcher, hdl]() { dispatcher.commit(hdl); });

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
              TAG_NEIGHBOR_ALLTOALL_CARTESIAN_BASE - 2 * dim,
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
              TAG_NEIGHBOR_ALLTOALL_CARTESIAN_BASE - 2 * dim - 1,
              comm.mpiComm()));
    }

    recvbuf = add(recvbuf, recvextent * recvcount);
  }

  /* then post sends */
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
              TAG_NEIGHBOR_ALLTOALL_CARTESIAN_BASE - 2 * dim - 1,
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
              TAG_NEIGHBOR_ALLTOALL_CARTESIAN_BASE - 2 * dim,
              comm.mpiComm()));
    }

    sendbuf = add(sendbuf, sendextent * sendcount);
  }

  return future;
}

collective_future NeighborAlltoallCtx::alltoall_dist_graph() {
  int indegree;
  int outdegree;
  int weighted;
  FMPI_CHECK_MPI(MPI_Dist_graph_neighbors_count(
      comm.mpiComm(), &indegree, &outdegree, &weighted));

  fmpi::FixedVector<int> sources(indegree);
  fmpi::FixedVector<int> destinations(outdegree);

  FMPI_CHECK_MPI(MPI_Dist_graph_neighbors(
      comm.mpiComm(),
      indegree,
      sources.data(),
      MPI_UNWEIGHTED,
      outdegree,
      destinations.data(),
      MPI_UNWEIGHTED));

  auto nslots = std::array<std::size_t, detail::n_types>{
      static_cast<std::size_t>(indegree),
      static_cast<std::size_t>(outdegree)};

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

  // setup dispatcher dispatcher
  auto&      dispatcher = static_dispatcher_pool();
  auto const hdl        = dispatcher.submit(std::move(schedule_state));
  auto       finalizer =
      rtlx::scope_exit([&dispatcher, hdl]() { dispatcher.commit(hdl); });

  for (auto&& src : sources) {
    dispatcher.schedule(
        hdl,
        message_type::IRECV,
        make_receive(
            recvbuf,
            recvcount,
            recvtype,
            static_cast<mpi::Rank>(src),
            TAG_NEIGHBOR_ALLTOALL_GRAPH,
            comm.mpiComm()));

    recvbuf = add(recvbuf, recvextent * recvcount);
  }

  for (auto&& dest : destinations) {
    dispatcher.schedule(
        hdl,
        message_type::ISEND,
        make_send(
            sendbuf,
            sendcount,
            sendtype,
            static_cast<mpi::Rank>(dest),
            TAG_NEIGHBOR_ALLTOALL_GRAPH,
            comm.mpiComm()));

    sendbuf = add(sendbuf, sendextent * sendcount);
  }

  return future;
}

collective_future NeighborAlltoallCtx::execute() {
  if (topology == MPI_CART) {
    return alltoall_cartesian();
  } else if (topology == MPI_DIST_GRAPH) {
    return alltoall_dist_graph();
  } else {
    throw NotSupportedException{};
  }
}

}  // namespace detail
}  // namespace fmpi
