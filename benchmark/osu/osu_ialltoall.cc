#include <omp.h>
#include <unistd.h>

#include <cstring>
#include <fmpi/Alltoall.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <iostream>
#include <numeric>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
#include <tlx/math/div_ceil.hpp>

#include "osu.hpp"

static constexpr std::size_t MIN_MESSAGE_SIZE = 1;
static constexpr uint32_t    COLL_LOOP_SMALL  = 1000;
static constexpr uint32_t    COLL_SKIP_SMALL  = 100;
static constexpr uint32_t    COLL_LOOP_LARGE  = 100;
static constexpr uint32_t    COLL_SKIP_LARGE  = 10;

double dummy_compute(double seconds) {
  double test_time = 0.0;

  test_time = do_compute_and_probe(seconds);

  return test_time;
}

fmpi::ScheduleOpts schedule_options(
    int                            algorithm,
    std::uint32_t                  winsz,
    std::string_view               id,
    fmpi::ScheduleOpts::WindowType win_type,
    mpi::Context const&            ctx);

int main(int argc, char* argv[]) {
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  if (!read_input(argc, argv)) {
    return 0;
  }

  // options.smin          = std::max(options.smin, MIN_MESSAGE_SIZE);
  // options.warmups       = std::max(options.warmups, COLL_SKIP_SMALL);
  // options.warmups_large = std::max(options.warmups_large, COLL_SKIP_LARGE);
  // options.iterations    = std::max(options.iterations, COLL_LOOP_SMALL);
  // options.iterations_large =
  //    std::max(options.iterations_large, COLL_LOOP_LARGE);

  const auto& world    = mpi::Context::world();
  auto const  rank     = world.rank();
  auto const  numprocs = world.size();

  if (numprocs < 2) {
    if (rank == 0) {
      fprintf(stderr, "This test requires at least two processes\n");
    }

    return EXIT_FAILURE;
  }

#if 0
  using value_t = int;
  auto mpi_type = MPI_INT;
#else
  using value_t = char;
  auto mpi_type = MPI_CHAR;
#endif

  auto const bufsize = options.smax * numprocs;

  value_t* sendbuf = NULL;
  value_t* recvbuf = NULL;

  if (allocate_memory_coll((void**)&sendbuf, bufsize * sizeof(value_t))) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
    world.abort(EXIT_FAILURE);
  }

#if 0
  std::iota(sendbuf, sendbuf + bufsize, world.rank() * bufsize);
#else
  std::memset(sendbuf, 1, bufsize);
#endif

  if (allocate_memory_coll((void**)&recvbuf, bufsize * sizeof(value_t))) {
    fprintf(
        stderr,
        "Could Not Allocate Memory [rank %d]\n",
        static_cast<int>(rank));
    world.abort(EXIT_FAILURE);
  }

  // std::memset(recvbuf, 0, bufsize);
  std::memset(recvbuf, 1, bufsize);

  print_preamble_nbc(rank, std::string_view("osu_ialltoall"));

  for (auto size = options.smin; size <= options.smax; size *= 2) {
    if (size > LARGE_MESSAGE_SIZE) {
      options.warmups    = options.warmups_large;
      options.iterations = options.iterations_large;
    }

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

    double timer   = 0.0;
    double t_start = 0.0;
    double t_stop  = 0.0;

    auto const     win_type = fmpi::ScheduleOpts::WindowType::fixed;
    constexpr auto winsz    = 64ul;
    auto const     opts =
        schedule_options(options.algorithm, winsz, "", win_type, world);
    for (uint32_t i = 0; i < options.iterations + options.warmups; i++) {
      t_start = MPI_Wtime();
#if 0
      auto future = fmpi::alltoall_tune(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, world);
#else
      auto future = fmpi::alltoall(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, world, opts);
#endif

      future.wait();
      FMPI_DBG_RANGE(recvbuf, recvbuf + bufsize);
      t_stop = MPI_Wtime();

      if (i >= options.warmups) {
        timer += t_stop - t_start;
      }
      MPI_CHECK(MPI_Barrier(world.mpiComm()));
    }

    FMPI_DBG("after warmups");
    // clear trace everything

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

    /* This is the pure comm. time */
    auto const latency = (timer * 1e6) / options.iterations;

    /* Comm. latency in seconds, fed to dummy_compute */
    auto const latency_in_secs = timer / options.iterations;

    FMPI_DBG(latency_in_secs);

    init_arrays(latency_in_secs);

    MPI_CHECK(MPI_Barrier(world.mpiComm()));

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
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, world);
#else
      auto future = fmpi::alltoall(
          sendbuf, size, mpi_type, recvbuf, size, mpi_type, world, opts);
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

fmpi::ScheduleOpts schedule_options(
    int                            algorithm,
    std::uint32_t                  winsz,
    std::string_view               id,
    fmpi::ScheduleOpts::WindowType win_type,
    mpi::Context const&            ctx) {
  if (algorithm == 0) {
    return fmpi::ScheduleOpts{fmpi::FlatHandshake{ctx}, winsz, "", win_type};
  } else if (algorithm == 1) {
    return fmpi::ScheduleOpts{fmpi::OneFactor{ctx}, winsz, "", win_type};
  }

  return fmpi::ScheduleOpts{fmpi::Bruck{ctx}, winsz, "", win_type};
}
