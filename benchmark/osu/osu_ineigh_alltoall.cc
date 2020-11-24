#include <omp.h>
#include <unistd.h>

#include <cstring>
#include <fmpi/Debug.hpp>
#include <fmpi/NeighborAlltoall.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <iostream>
#include <numeric>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
#include <tlx/math/div_ceil.hpp>

#include "osu.hpp"

double dummy_compute(double seconds) {
  double test_time = 0.0;

  test_time = do_compute_and_probe(seconds);

  return test_time;
}

constexpr int ndims   = 3;
constexpr int periods = 0;

mpi::Context make_cartesian_comm(mpi::Context const& old_comm) {
  MPI_Comm               comm_cube{};
  std::array<int, ndims> cart_dimensions{};
  std::array<int, ndims> cart_periodicity{};
  cart_periodicity.fill(periods);

  MPI_Dims_create(old_comm.size(), ndims, cart_dimensions.data());

  FMPI_DBG(cart_dimensions);

  MPI_Cart_create(
      old_comm.mpiComm(),
      ndims,
      cart_dimensions.data(),
      cart_periodicity.data(),
      periods,
      &comm_cube);

  return mpi::Context{comm_cube};
}

int main(int argc, char* argv[]) {
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  if (!read_input(argc, argv)) {
    return 0;
  }

  const auto& world    = mpi::Context::world();
  auto const  rank     = world.rank();
  auto const  numprocs = world.size();

  mpi::Context mycomm = make_cartesian_comm(world);

  if (numprocs < 2) {
    if (rank == 0) {
      fprintf(stderr, "This test requires at least two processes\n");
    }

    return EXIT_FAILURE;
  }

#if FMPI_DEBUG_ASSERT
  using value_t = int;
  auto mpi_type = MPI_INT;
#pragma message( \
    "WARNING: You are compiling in debug mode. Be careful with benchmarks.")
#else
  using value_t = char;
  auto mpi_type = MPI_CHAR;
#endif

  auto const bufsize = options.smax * numprocs;

  value_t* sendbuf     = NULL;
  value_t* recvbuf     = NULL;
  value_t* recvbuf_mpi = NULL;

  if (allocate_memory_coll((void**)&sendbuf, bufsize * sizeof(value_t))) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank.mpiRank());
    world.abort(EXIT_FAILURE);
  }

#if FMPI_DEBUG_ASSERT
  std::iota(sendbuf, sendbuf + bufsize, mycomm.rank() * bufsize);
#else
  std::memset(sendbuf, 1, bufsize * sizeof(value_t));
#endif

  if (allocate_memory_coll((void**)&recvbuf, bufsize * sizeof(value_t))) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank.mpiRank());
    world.abort(EXIT_FAILURE);
  }

  if (allocate_memory_coll((void**)&recvbuf_mpi, bufsize * sizeof(value_t))) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank.mpiRank());
    world.abort(EXIT_FAILURE);
  }

  std::memset(recvbuf, 0, bufsize);

  {
    auto const size = options.smax;
    MPI_Neighbor_alltoall(
        sendbuf,
        size,
        mpi_type,
        recvbuf_mpi,
        size,
        mpi_type,
        mycomm.mpiComm());
    auto future = fmpi::neighbor_alltoall(
        sendbuf, size, mpi_type, recvbuf, size, mpi_type, mycomm);
    FMPI_DBG_RANGE(recvbuf_mpi, recvbuf_mpi + bufsize);
    future.wait();
    FMPI_DBG_RANGE(recvbuf, recvbuf + bufsize);
  }

  if (not std::equal(recvbuf_mpi, recvbuf_mpi + options.smax, recvbuf)) {
    throw std::runtime_error("invalid result");
  }

  return 0;

  print_preamble_nbc(rank, std::string_view("osu_ialltoall"));

  for (auto size = options.smin; size <= options.smax; size *= 2) {
    if (size > LARGE_MESSAGE_SIZE) {
      options.warmups    = options.warmups_large;
      options.iterations = options.iterations_large;
    }

    MPI_CHECK(MPI_Barrier(mycomm.mpiComm()));

    double timer   = 0.0;
    double t_start = 0.0;
    double t_stop  = 0.0;

    for (uint32_t i = 0; i < options.iterations + options.warmups; i++) {
      t_start = MPI_Wtime();
#if 0
      auto future = fmpi::alltoall_tune(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, mycomm);
#else
      auto future = fmpi::neighbor_alltoall(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, mycomm);

      return 0;
#endif

      future.wait();
      FMPI_DBG_RANGE(recvbuf, recvbuf + bufsize);
      t_stop = MPI_Wtime();

      if (i >= options.warmups) {
        timer += t_stop - t_start;
      }
      MPI_CHECK(MPI_Barrier(mycomm.mpiComm()));
    }

    FMPI_DBG("after warmups");
    // clear trace everything

    MPI_CHECK(MPI_Barrier(mycomm.mpiComm()));

    /* This is the pure comm. time */
    auto const latency = (timer * 1e6) / options.iterations;

    /* Comm. latency in seconds, fed to dummy_compute */
    auto const latency_in_secs = timer / options.iterations;

    FMPI_DBG(latency_in_secs);

    init_arrays(latency_in_secs);

    MPI_CHECK(MPI_Barrier(mycomm.mpiComm()));

    timer              = 0.0;
    double tcomp_total = 0.0;
    double init_total  = 0.0;
    double wait_total  = 0.0;
    double test_total  = 0.0;

    for (uint32_t i = 0; i < options.iterations + options.warmups; i++) {
      t_start = MPI_Wtime();

      auto init_time = MPI_Wtime();

#if 0
      auto future = fmpi::alltoall_tune(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, mycomm);
#else
      auto future = fmpi::neighbor_alltoall(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, mycomm);
#endif

      init_time = MPI_Wtime() - init_time;

      auto tcomp     = MPI_Wtime();
      auto test_time = dummy_compute(latency_in_secs);
      tcomp          = MPI_Wtime() - tcomp;

      auto wait_time = MPI_Wtime();
      future.wait();
      wait_time = MPI_Wtime() - wait_time;

      t_stop = MPI_Wtime();

      if (i == options.warmups) {
        fmpi::TraceStore::instance().erase(std::string_view("schedule_ctx"));
        assert(fmpi::TraceStore::instance().empty());
      }

      if (i >= options.warmups) {
        timer += t_stop - t_start;
        tcomp_total += tcomp;
        init_total += init_time;
        test_total += test_time;
        wait_total += wait_time;
      }

      MPI_CHECK(MPI_Barrier(mycomm.mpiComm()));
    }

    MPI_Barrier(mycomm.mpiComm());

    calculate_and_print_stats(
        rank,
        size,
        numprocs,
        timer,
        latency,
        test_total,
        tcomp_total,
        wait_total,
        init_total,
        mycomm,
        1);
  }

  free_buffer(sendbuf);
  free_buffer(recvbuf);

  return EXIT_SUCCESS;
}
