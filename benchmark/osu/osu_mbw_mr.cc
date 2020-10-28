#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <iostream>
#include <rtlx/ScopedLambda.hpp>

#include "osu.hpp"

MPI_Request* mbw_request;
MPI_Status*  mbw_reqstat;

double calc_bw(
    int                 rank,
    int                 size,
    int                 num_pairs,
    int                 window_size,
    char*               s_buf,
    char*               r_buf,
    mpi::Context const& comm);

void perform_work(
    int                 rank,
    int                 size,
    int                 num_pairs,
    int                 window_size,
    char*               s_buf,
    char*               r_buf,
    mpi::Context const& comm,
    int                 n);

#define BW_LOOP_SMALL 100
#define BW_SKIP_SMALL 10
#define BW_LOOP_LARGE 20
#define BW_SKIP_LARGE 2

int main(int argc, char* argv[]) {
  char *s_buf, *r_buf;
  // Our value type
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  const auto& world = mpi::Context::world();

  if (world.size() < 2) {
    if (world.rank() == 0) {
      fprintf(stderr, "This test requires at least two processes\n");
    }

    return EXIT_FAILURE;
  }

  if (!read_input(argc, argv)) {
    return 0;
  }

  int wait = 0;
  while (wait)
    ;

  options.smin  = std::max(options.smin, 1ul);
  options.pairs = world.size() / 2;
  // options.iterations       = BW_LOOP_SMALL;
  // options.skip             = BW_SKIP_SMALL;
  // options.iterations_large = BW_LOOP_LARGE;
  // options.skip_large       = BW_SKIP_LARGE;

  print_topology(world, num_nodes(world));

  MPI_Barrier(world.mpiComm());

  if (allocate_memory_pt2pt_mul(
          &s_buf, &r_buf, world.rank(), options.pairs)) {
    /* Error allocating memory */
    return EXIT_FAILURE;
  }

  assert(options.window_varied);

  auto const rank = world.rank();

  if (world.rank() == 0) {
    fprintf(
        stdout, "# [ pairs: %d ] [ window size: varied ]\n", options.pairs);
    fprintf(stdout, "\n# Uni-directional Bandwidth (MB/sec)\n");

    fflush(stdout);
  }

  int      window_array[] = WINDOW_SIZES;
  double** bandwidth_results;
  int      log_val = 1, tmp_message_size = options.smax;
  int      i, j;

  for (i = 0; i < WINDOW_SIZES_COUNT; i++) {
    if (window_array[i] > options.window_size) {
      options.window_size = window_array[i];
    }
  }

  mbw_request =
      (MPI_Request*)malloc(sizeof(MPI_Request) * options.window_size);
  mbw_reqstat = (MPI_Status*)malloc(sizeof(MPI_Status) * options.window_size);

  while (tmp_message_size >>= 1) {
    log_val++;
  }

  bandwidth_results = (double**)malloc(sizeof(double*) * log_val);

  for (i = 0; i < log_val; i++) {
    bandwidth_results[i] =
        (double*)malloc(sizeof(double) * WINDOW_SIZES_COUNT);
  }

  if (rank == 0) {
    fprintf(stdout, "#      ");

    for (i = 0; i < WINDOW_SIZES_COUNT; i++) {
      fprintf(stdout, "  %10d", window_array[i]);
    }

    fprintf(stdout, "\n");
    fflush(stdout);
  }

  std::size_t curr_size = options.smin;
  int         c;

  for (j = 0; curr_size <= options.smax; curr_size *= 2, j++) {
    if (rank == 0) {
      fprintf(stdout, "%-7d", curr_size);
    }

    for (i = 0; i < WINDOW_SIZES_COUNT; i++) {
      bandwidth_results[j][i] = calc_bw(
          rank,
          curr_size,
          options.pairs,
          window_array[i],
          s_buf,
          r_buf,
          world);

      if (rank == 0) {
        fprintf(stdout, "  %10.*f", FLOAT_PRECISION, bandwidth_results[j][i]);
      }
    }

    if (rank == 0) {
      fprintf(stdout, "\n");
      fflush(stdout);
    }
  }

  if (rank == 0 && true /*options.print_rate*/) {
    fprintf(stdout, "\n# Message Rate Profile\n");
    fprintf(stdout, "#      ");

    for (i = 0; i < WINDOW_SIZES_COUNT; i++) {
      fprintf(stdout, "  %10d", window_array[i]);
    }

    fprintf(stdout, "\n");
    fflush(stdout);

    for (c = 0, curr_size = options.smin; curr_size <= options.smax;
         curr_size *= 2) {
      fprintf(stdout, "%-7d", curr_size);

      for (i = 0; i < WINDOW_SIZES_COUNT; i++) {
        double rate = 1e6 * bandwidth_results[c][i] / curr_size;

        fprintf(stdout, "  %10.2f", rate);
      }

      fprintf(stdout, "\n");
      fflush(stdout);
      c++;
    }
  }

  free_memory_pt2pt_mul(s_buf, r_buf, rank, options.pairs);

  return EXIT_SUCCESS;
}

void send_messages(
    int                   rank,
    int                   size,
    int                   num_pairs,
    int                   window_size,
    char*                 s_buf,
    char*                 r_buf,
    mpi::Context const&   comm,
    int                   target,
    fmpi::ScheduleHandle  hdl,
    fmpi::CommDispatcher& dispatcher);

void recv_messages(
    int                   rank,
    int                   size,
    int                   num_pairs,
    int                   window_size,
    char*                 s_buf,
    char*                 r_buf,
    mpi::Context const&   ctx,
    int                   target,
    fmpi::ScheduleHandle  hdl,
    fmpi::CommDispatcher& dispatcher);

double calc_bw(
    int                 rank,
    int                 size,
    int                 num_pairs,
    int                 window_size,
    char*               s_buf,
    char*               r_buf,
    mpi::Context const& ctx) {
  double t_start = 0, t_end = 0, t = 0, sum_time = 0, bw = 0;
  int    i, j, target;

  set_buffer_pt2pt(s_buf, rank, 'a', size);
  set_buffer_pt2pt(r_buf, rank, 'b', size);

  MPI_CHECK(MPI_Barrier(ctx.mpiComm()));

  std::array<std::size_t, 2> slots{0, 0};

  if (rank < num_pairs) {
    slots[0] = 1;
    slots[1] = window_size;
  } else if (rank < num_pairs * 2) {
    slots[0] = window_size;
    slots[1] = 1;
  }

  auto& dispatcher = fmpi::static_dispatcher_pool();
  if (rank < num_pairs) {
    target = rank + num_pairs;

    {
      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      for (i = 0; i < options.warmups; i++) {
        send_messages(
            rank,
            size,
            num_pairs,
            window_size,
            s_buf,
            r_buf,
            ctx,
            target,
            hdl,
            dispatcher);
      }
      dispatcher.commit(hdl);
    }

    MPI_CHECK(MPI_Barrier(ctx.mpiComm()));

    FMPI_DBG("after warump");

    auto const max_msgs = (window_size + 1) * options.iterations;

    auto promise        = fmpi::collective_promise{};
    auto future         = promise.get_future();
    auto schedule_state = std::make_unique<fmpi::ScheduleCtx>(
        slots, std::move(promise), max_msgs);

    // submit into dispatcher
    auto const hdl = dispatcher.submit(std::move(schedule_state));

    t_start = MPI_Wtime();
    for (i = 0; i < options.iterations; i++) {
      send_messages(
          rank,
          size,
          num_pairs,
          window_size,
          s_buf,
          r_buf,
          ctx,
          target,
          hdl,
          dispatcher);
    }

    dispatcher.commit(hdl);
    future.wait();
    t_end = MPI_Wtime();
    t     = t_end - t_start;
  } else if (rank < num_pairs * 2) {
    target = rank - num_pairs;
#if 1
    {
      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      for (i = 0; i < options.warmups; i++) {
        recv_messages(
            rank,
            size,
            num_pairs,
            window_size,
            s_buf,
            r_buf,
            ctx,
            target,
            hdl,
            dispatcher);
      }
      dispatcher.commit(hdl);
    }

    MPI_CHECK(MPI_Barrier(ctx.mpiComm()));

    FMPI_DBG("after warump");

    auto const max_msgs = (window_size + 1) * options.iterations;

    auto promise        = fmpi::collective_promise{};
    auto future         = promise.get_future();
    auto schedule_state = std::make_unique<fmpi::ScheduleCtx>(
        slots, std::move(promise), max_msgs);

    // submit into dispatcher
    auto const hdl = dispatcher.submit(std::move(schedule_state));

    for (i = 0; i < options.iterations; i++) {
      recv_messages(
          rank,
          size,
          num_pairs,
          window_size,
          s_buf,
          r_buf,
          ctx,
          target,
          hdl,
          dispatcher);
    }

    dispatcher.commit(hdl);

#else
    for (i = 0; i < options.iterations + options.warmups; i++) {
      if (i == options.warmups) {
        MPI_CHECK(MPI_Barrier(ctx.mpiComm()));
      }

      for (j = 0; j < window_size; j++) {
        MPI_CHECK(MPI_Irecv(
            r_buf,
            size,
            MPI_CHAR,
            target,
            100,
            ctx.mpiComm(),
            mbw_request + j));
      }

      MPI_CHECK(MPI_Waitall(window_size, mbw_request, mbw_reqstat));
      MPI_CHECK(MPI_Send(s_buf, 4, MPI_CHAR, target, 101, ctx.mpiComm()));
    }
#endif
  }

  else {
    MPI_CHECK(MPI_Barrier(ctx.mpiComm()));
  }

  // MPI_CHECK(MPI_Barrier(ctx.mpiComm()));

  MPI_CHECK(
      MPI_Reduce(&t, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));

  if (rank == 0) {
    double tmp = size / 1e6 * num_pairs;

    sum_time /= num_pairs;
    tmp = tmp * options.iterations * window_size;
    bw  = tmp / sum_time;

    return bw;
  }

  return 0;
}

void send_messages(
    int                   rank,
    int                   size,
    int                   num_pairs,
    int                   window_size,
    char*                 s_buf,
    char*                 r_buf,
    mpi::Context const&   comm,
    int                   target,
    fmpi::ScheduleHandle  hdl,
    fmpi::CommDispatcher& dispatcher) {
  FMPI_DBG(hdl.id());
  FMPI_DBG(window_size);
  for (int j = 0; j < window_size; j++) {
    // MPI_CHECK(MPI_Isend(
    //     s_buf,
    //     size,
    //     MPI_CHAR,
    //     target,
    //     100,
    //     ctx.mpiComm(),
    //     mbw_request + j));
    auto send = fmpi::make_send(
        s_buf, size, MPI_CHAR, mpi::Rank{target}, 100, comm.mpiComm());

    dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);
  }
  // MPI_CHECK(MPI_Waitall(window_size, mbw_request, mbw_reqstat));
  dispatcher.schedule(hdl, fmpi::message_type::BARRIER);

  // MPI_CHECK(MPI_Recv(
  //    r_buf, 4, MPI_CHAR, target, 101, ctx.mpiComm(), &mbw_reqstat[0]));
  auto recv = fmpi::make_receive(
      r_buf, 4, MPI_CHAR, mpi::Rank{target}, 101, comm.mpiComm());
  dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);
}

void recv_messages(
    int                   rank,
    int                   size,
    int                   num_pairs,
    int                   window_size,
    char*                 s_buf,
    char*                 r_buf,
    mpi::Context const&   ctx,
    int                   target,
    fmpi::ScheduleHandle  hdl,
    fmpi::CommDispatcher& dispatcher) {
#if 0
    if (i == options.warmups) {
      MPI_CHECK(MPI_Barrier(ctx.mpiComm()));
    }

    for (j = 0; j < window_size; j++) {
      MPI_CHECK(MPI_Irecv(
          r_buf,
          size,
          MPI_CHAR,
          target,
          100,
          ctx.mpiComm(),
          mbw_request + j));
    }

    MPI_CHECK(MPI_Waitall(window_size, mbw_request, mbw_reqstat));
    MPI_CHECK(MPI_Send(s_buf, 4, MPI_CHAR, target, 101, ctx.mpiComm()));
#endif

  for (int j = 0; j < window_size; j++) {
    auto recv = fmpi::make_receive(
        r_buf, size, MPI_CHAR, mpi::Rank{target}, 100, ctx.mpiComm());

    dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);
  }
  // MPI_CHECK(MPI_Waitall(window_size, mbw_request, mbw_reqstat));
  dispatcher.schedule(hdl, fmpi::message_type::BARRIER);

  // MPI_CHECK(MPI_Recv(
  //    r_buf, 4, MPI_CHAR, target, 101, ctx.mpiComm(), &mbw_reqstat[0]));
  auto send = fmpi::make_send(
      s_buf, 4, MPI_CHAR, mpi::Rank{target}, 101, ctx.mpiComm());
  dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);
}
