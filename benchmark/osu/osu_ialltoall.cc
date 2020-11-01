#include <omp.h>
#include <unistd.h>

#include <cstring>
#include <fmpi/Alltoall.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <iostream>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
#include <tlx/math/div_ceil.hpp>

#include "osu.hpp"

double dummy_compute(double seconds) {
  double test_time = 0.0;

  test_time = do_compute_and_probe(seconds);

  return test_time;
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

  if (numprocs < 2) {
    if (rank == 0) {
      fprintf(stderr, "This test requires at least two processes\n");
    }

    return EXIT_FAILURE;
  }

  auto const bufsize = options.smax * numprocs;

  char* sendbuf = NULL;
  char* recvbuf = NULL;

  if (allocate_memory_coll((void**)&sendbuf, bufsize)) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
    world.abort(EXIT_FAILURE);
  }

  std::memset(sendbuf, 1, bufsize);

  if (allocate_memory_coll((void**)&recvbuf, options.smax * numprocs)) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
    world.abort(EXIT_FAILURE);
  }

  std::memset(recvbuf, 1, bufsize);

  print_preamble_nbc(rank, std::string_view("osu_ialltoall"));

  double timer = 0.0;
  double t_start, t_stop;

  for (auto size = options.smin; size <= options.smax; size *= 2) {
    if (size > LARGE_MESSAGE_SIZE) {
      options.warmups    = options.warmups_large;
      options.iterations = options.iterations_large;
    }

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

    auto           schedule = fmpi::FlatHandshake{world};
    auto const     win_type = fmpi::ScheduleOpts::WindowType::fixed;
    constexpr auto winsz    = 64ul;
    auto const     opts = fmpi::ScheduleOpts{schedule, winsz, "", win_type};
    for (auto i = 0; i < options.iterations + options.warmups; i++) {
      auto future = fmpi::alltoall(
          sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, world, opts);

      future.wait();
      t_stop = MPI_Wtime();

      if (i >= options.warmups) {
        timer += t_stop - t_start;
      }
      MPI_CHECK(MPI_Barrier(world.mpiComm()));
    }

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

    /* This is the pure comm. time */
    auto const latency = (timer * 1e6) / options.iterations;

    /* Comm. latency in seconds, fed to dummy_compute */
    auto const latency_in_secs = timer / options.iterations;

    init_arrays(latency_in_secs);

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

    timer              = 0.0;
    double tcomp_total = 0.0;
    double init_total  = 0.0;
    double wait_total  = 0.0;
    double test_total  = 0.0;

    for (auto i = 0; i < options.iterations + options.warmups; i++) {
      t_start = MPI_Wtime();

      auto init_time = MPI_Wtime();

      auto future = fmpi::alltoall(
          sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, world, opts);

      init_time = MPI_Wtime() - init_time;

      auto tcomp     = MPI_Wtime();
      auto test_time = dummy_compute(latency_in_secs);
      tcomp          = MPI_Wtime() - tcomp;

      auto wait_time = MPI_Wtime();
      future.wait();
      wait_time = MPI_Wtime() - wait_time;

      t_stop = MPI_Wtime();

      if (i >= options.warmups) {
        timer += t_stop - t_start;
        tcomp_total += tcomp;
        init_total += init_time;
        test_total += test_time;
        wait_total += wait_time;
      }
      MPI_CHECK(MPI_Barrier(world.mpiComm()));
    }

    MPI_Barrier(world.mpiComm());

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
        world);
  }

  free_buffer(sendbuf);
  free_buffer(recvbuf);

  return EXIT_SUCCESS;
}
