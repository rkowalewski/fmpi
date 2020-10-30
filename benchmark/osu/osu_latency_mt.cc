#include <unistd.h>

#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <iostream>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
#include <tlx/math/div_ceil.hpp>

#include "osu.hpp"

pthread_barrier_t sender_barrier;
pthread_barrier_t recv_barrier;

// static int num_threads_sender = 1;

int    finished_size        = 0;
int    finished_size_sender = 0;
double t_start = 0, t_end = 0;

pthread_mutex_t finished_size_mutex;
pthread_cond_t  finished_size_cond;
pthread_mutex_t finished_size_sender_mutex;
pthread_cond_t  finished_size_sender_cond;

constexpr std::size_t MAX_NUM_THREADS = 128;
constexpr std::size_t MAX_REQ_NUM     = 1000;

struct thread_tag {
  int                   id{};
  fmpi::CommDispatcher* static_dispatcher;
};

#define MYDBG(msg) FMPI_DBG_STREAM("thread " << thread_id->id << ": " << msg)

using thread_tag_t = thread_tag;

// MPI_Status reqstat[MAX_REQ_NUM];

void* send_thread(void* arg);
void* recv_thread(void* arg);

void* test(void* arg) {
  const auto& world = mpi::Context::world();
  printf("hello %d out of %d\n", ((thread_tag_t*)arg)->id, world.size());
  return 0;
}

int main(int argc, char* argv[]) {
  std::vector<thread_tag_t> tags{MAX_NUM_THREADS};
  std::vector<pthread_t>    sr_threads(MAX_NUM_THREADS);
  // Our value type
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Multiple);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  const auto& world = mpi::Context::world();

  auto const myid = world.rank();

  if (world.size() != 2) {
    if (myid == 0) {
      fprintf(stderr, "This test requires exactly two processes\n");
    }

    return EXIT_FAILURE;
  }

  if (!read_input(argc, argv)) {
    return 0;
  }

  if (options.sender_threads == -1) {
    options.sender_threads = 1;
  }

  finished_size        = 1;
  finished_size_sender = 1;

  print_topology(world, num_nodes(world));

  pthread_barrier_init(&sender_barrier, NULL, options.sender_threads);
  pthread_barrier_init(&recv_barrier, NULL, options.num_threads);

  MPI_Barrier(world.mpiComm());

  printf("after barrier\n");

  auto& mydispatcher = fmpi::static_dispatcher_pool();

  if (myid == 0) {
    printf(
        "# Number of Sender threads: %d \n# Number of Receiver threads: %d\n",
        options.sender_threads,
        options.num_threads);

    auto const nt = options.sender_threads;

    for (auto i = 0; i < nt; i++) {
      tags[i].id                = i;
      tags[i].static_dispatcher = &mydispatcher;
      pthread_create(&sr_threads[i], NULL, send_thread, &tags[i]);
    }
    for (auto i = 0; i < nt; i++) {
      pthread_join(sr_threads[i], NULL);
      FMPI_DBG_STREAM("thread " << i << " joined");
    }
  } else {
    for (auto i = 0; i < options.num_threads; i++) {
      tags[i].id                = i;
      tags[i].static_dispatcher = &mydispatcher;
      pthread_create(&sr_threads[i], NULL, recv_thread, &tags[i]);
    }

    for (auto i = 0; i < options.num_threads; i++) {
      pthread_join(sr_threads[i], NULL);
      FMPI_DBG_STREAM("thread " << i << " joined");
    }
  }

  return EXIT_SUCCESS;
}

void* send_thread(void* arg) {
  unsigned long align_size = sysconf(_SC_PAGESIZE);
  int           i = 0, val = 0, iter = 0;
  char *        s_buf, *r_buf;
  double        t = 0, latency = 0;
  thread_tag_t* thread_id = (thread_tag_t*)arg;
  char*         ret       = NULL;

  val = thread_id->id;

  MYDBG("enter");

  const auto& world = mpi::Context::world();

  int const myid = world.rank();

  if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
    /* Error allocating memory */
    fprintf(
        stderr,
        "Error allocating memory on Rank %d, thread ID %d\n",
        myid,
        thread_id->id);
    //*ret = '1';
    return (void*)1;
  }

  MYDBG("allocated memory");

  auto& dispatcher = *(thread_id->static_dispatcher);
  MYDBG("got dispatcher");

  for (std::size_t size = options.smin, iter = 0; size <= options.smax;
       size = (size ? size * 2 : 1)) {
    pthread_mutex_lock(&finished_size_sender_mutex);

    MYDBG(options.sender_threads);

    if (finished_size_sender == options.sender_threads) {
      MPI_CHECK(MPI_Barrier(world.mpiComm()));

      finished_size_sender = 1;

      pthread_mutex_unlock(&finished_size_sender_mutex);
      pthread_cond_broadcast(&finished_size_sender_cond);
    } else {
      finished_size_sender++;

      pthread_cond_wait(
          &finished_size_sender_cond, &finished_size_sender_mutex);
      pthread_mutex_unlock(&finished_size_sender_mutex);
    }

    MYDBG("reached barrier");

    if (size > LARGE_MESSAGE_SIZE) {
      options.iterations = options.iterations_large;
      options.warmups    = options.warmups_large;
    }

    /* touch the data */
    set_buffer_pt2pt(s_buf, myid, 'a', size);
    set_buffer_pt2pt(r_buf, myid, 'b', size);

    {
      MYDBG("preparing dispatcher");
      auto slots = std::array<std::size_t, 2>{};
      slots.fill(tlx::div_ceil(options.warmups, options.sender_threads));

      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));

      // submit into dispatcher
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      MYDBG(hdl.id());

      for (i = val; i < options.warmups; i += options.sender_threads) {
        auto const s_tag = (options.sender_threads > 1) ? i : 1;
        auto const r_tag = (options.sender_threads > 1) ? i : 2;

        FMPI_DBG(std::make_tuple(thread_id->id, hdl.id(), s_tag, r_tag));
        // MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, world.mpiComm()));
        // MPI_CHECK(MPI_Recv(
        //    r_buf, size, MPI_CHAR, 1, i, world.mpiComm(), &reqstat[val]));
        auto send = fmpi::make_send(
            s_buf, size, MPI_CHAR, mpi::Rank{1}, s_tag, world.mpiComm());
        auto recv = fmpi::make_receive(
            r_buf, size, MPI_CHAR, mpi::Rank{1}, r_tag, world.mpiComm());

        auto const s_succ =
            dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);

        auto const r_succ =
            dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);

        FMPI_ASSERT(s_succ and r_succ);
      }
      // guarantee that all threads have scheduled their requests
      pthread_barrier_wait(&sender_barrier);

      if (thread_id->id == 0) {
        // this acts as a kind of synchronization from threads to the
        // dispatcher
        dispatcher.schedule(hdl, fmpi::message_type::COMMIT_ALL);
      }
    }  // all threads wait until the future is ready

    MYDBG("after warmups");

    {
      MYDBG("preparing dispatcher");
      auto slots = std::array<std::size_t, 2>{};
      slots.fill(tlx::div_ceil(options.iterations, options.sender_threads));

      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));

      // submit into dispatcher
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      MYDBG(hdl.id());

      t_start = MPI_Wtime();

      for (i = val; i < options.iterations; i += options.sender_threads) {
        auto const s_tag = (options.sender_threads > 1) ? i : 1;
        auto const r_tag = (options.sender_threads > 1) ? i : 2;

        FMPI_DBG(std::make_tuple(thread_id->id, hdl.id(), s_tag, r_tag));
        // MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, world.mpiComm()));
        // MPI_CHECK(MPI_Recv(
        //    r_buf, size, MPI_CHAR, 1, i, world.mpiComm(), &reqstat[val]));
        auto send = fmpi::make_send(
            s_buf, size, MPI_CHAR, mpi::Rank{1}, s_tag, world.mpiComm());
        auto recv = fmpi::make_receive(
            r_buf, size, MPI_CHAR, mpi::Rank{1}, r_tag, world.mpiComm());

        auto const s_succ =
            dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);

        auto const r_succ =
            dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);

        FMPI_ASSERT(s_succ and r_succ);
      }
      // guarantee that all threads have scheduled their requests
      pthread_barrier_wait(&sender_barrier);

      if (thread_id->id == 0) {
        // this acts as a kind of synchronization from threads to the
        // dispatcher
        dispatcher.schedule(hdl, fmpi::message_type::COMMIT_ALL);
      }
    }  // all threads wait until the future is ready

    MYDBG("after iterations");

    t_end = MPI_Wtime();

#if 0
    for (i = val; i < options.iterations + options.warmups;
         i += options.sender_threads) {
      if (i == options.warmups) {
        t_start    = MPI_Wtime();
        flag_print = 1;
      }

      if (options.sender_threads > 1) {
        MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, world.mpiComm()));
        MPI_CHECK(MPI_Recv(
            r_buf, size, MPI_CHAR, 1, i, world.mpiComm(), &reqstat[val]));
      } else {
        MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, 1, world.mpiComm()));
        MPI_CHECK(MPI_Recv(
            r_buf, size, MPI_CHAR, 1, 2, world.mpiComm(), &reqstat[val]));
      }
    }
#endif

    int flag_print = thread_id->id == 0;

    if (flag_print == 1) {
      t = t_end - t_start;

      latency =
          (t)*1.0e6 / (2.0 * options.iterations / options.sender_threads);
      fprintf(
          stdout,
          "%-*d%*.*f\n",
          10,
          size,
          FIELD_WIDTH,
          FLOAT_PRECISION,
          latency);
      fflush(stdout);
    }
    iter++;
  }

  free_memory(s_buf, r_buf, myid);

  return 0;
}

void* recv_thread(void* arg) {
  unsigned long align_size = sysconf(_SC_PAGESIZE);
  int           i = 0, val = 0;
  int           iter = 0;
  char*         ret  = NULL;
  char *        s_buf, *r_buf;
  thread_tag_t* thread_id;

  thread_id = (thread_tag_t*)arg;
  val       = thread_id->id;

  MYDBG("enter");

  const auto& world = mpi::Context::world();

  int const myid = world.rank();

  if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
    /* Error allocating memory */
    fprintf(
        stderr,
        "Error allocating memory on Rank %d, thread ID %d\n",
        myid,
        thread_id->id);
    return (void*)1;
  }

  MYDBG("allocated memory");

  auto& dispatcher = *(thread_id->static_dispatcher);

  MYDBG("got dispatcher");

  for (std::size_t size = options.smin, iter = 0; size <= options.smax;
       size = (size ? size * 2 : 1)) {
    pthread_mutex_lock(&finished_size_mutex);

    if (finished_size == options.num_threads) {
      MPI_CHECK(MPI_Barrier(world.mpiComm()));

      finished_size = 1;

      pthread_mutex_unlock(&finished_size_mutex);
      pthread_cond_broadcast(&finished_size_cond);
    }

    else {
      finished_size++;

      pthread_cond_wait(&finished_size_cond, &finished_size_mutex);
      pthread_mutex_unlock(&finished_size_mutex);
    }

    MYDBG("reached barrier");

    if (size > LARGE_MESSAGE_SIZE) {
      options.iterations = options.iterations_large;
      options.warmups    = options.warmups_large;
    }

    /* touch the data */
    set_buffer_pt2pt(s_buf, myid, 'a', size);
    set_buffer_pt2pt(r_buf, myid, 'b', size);

    MYDBG("preparing schedule");

    {
      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto slots   = std::array<std::size_t, 2>{};
      slots.fill(tlx::div_ceil(options.warmups, options.num_threads));

      FMPI_DBG(slots);
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));

      // submit into dispatcher
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      MYDBG(hdl.id());

      for (i = val; i < options.warmups; i += options.num_threads) {
        auto const s_tag = (options.sender_threads > 1) ? i : 2;
        auto const r_tag = (options.sender_threads > 1) ? i : 1;

        FMPI_DBG(std::make_tuple(thread_id->id, hdl.id(), s_tag, r_tag));
        // MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, world.mpiComm()));
        // MPI_CHECK(MPI_Recv(
        //    r_buf, size, MPI_CHAR, 1, i, world.mpiComm(), &reqstat[val]));
        auto send = fmpi::make_send(
            s_buf, size, MPI_CHAR, mpi::Rank{0}, s_tag, world.mpiComm());
        auto recv = fmpi::make_receive(
            r_buf, size, MPI_CHAR, mpi::Rank{0}, r_tag, world.mpiComm());

        auto const r_succ =
            dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);

        auto const s_succ =
            dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);

        auto const b_succ = true;
        //   dispatcher.schedule(hdl, fmpi::message_type::BARRIER);

        FMPI_ASSERT(s_succ and r_succ and b_succ);
      }

      pthread_barrier_wait(&recv_barrier);

      if (thread_id->id == 0) {
        dispatcher.schedule(hdl, fmpi::message_type::COMMIT_ALL);
      }
    }  // we automatically wait for the future

    MYDBG("after warmups");

    {
      auto promise = fmpi::collective_promise{};
      auto future  = promise.get_future();
      auto slots   = std::array<std::size_t, 2>{};
      slots.fill(tlx::div_ceil(options.iterations, options.num_threads));

      FMPI_DBG(slots);
      auto schedule_state =
          std::make_unique<fmpi::ScheduleCtx>(slots, std::move(promise));

      // submit into dispatcher
      auto const hdl = dispatcher.submit(std::move(schedule_state));
      MYDBG(hdl.id());

      for (i = val; i < options.iterations; i += options.num_threads) {
        auto const s_tag = (options.sender_threads > 1) ? i : 2;
        auto const r_tag = (options.sender_threads > 1) ? i : 1;

        FMPI_DBG(std::make_tuple(thread_id->id, hdl.id(), s_tag, r_tag));
        // MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, world.mpiComm()));
        // MPI_CHECK(MPI_Recv(
        //    r_buf, size, MPI_CHAR, 1, i, world.mpiComm(), &reqstat[val]));
        auto send = fmpi::make_send(
            s_buf, size, MPI_CHAR, mpi::Rank{0}, s_tag, world.mpiComm());
        auto recv = fmpi::make_receive(
            r_buf, size, MPI_CHAR, mpi::Rank{0}, r_tag, world.mpiComm());

        auto const r_succ =
            dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);

        auto const s_succ =
            dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);

        auto const b_succ = true;
        //   dispatcher.schedule(hdl, fmpi::message_type::BARRIER);

        FMPI_ASSERT(s_succ and r_succ and b_succ);
      }

      pthread_barrier_wait(&recv_barrier);

      if (thread_id->id == 0) {
        dispatcher.schedule(hdl, fmpi::message_type::COMMIT_ALL);
      }
    }  // we automatically wait for the future

    MYDBG("after iterations");

#if 0

    for (i = val; i < (options.iterations + options.warmups);
         i += options.num_threads) {
      if (options.sender_threads > 1) {
        MPI_Recv(r_buf, size, MPI_CHAR, 0, i, world.mpiComm(), &reqstat[val]);
        MPI_Send(s_buf, size, MPI_CHAR, 0, i, world.mpiComm());
      } else {
        MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, world.mpiComm(), &reqstat[val]);
        MPI_Send(s_buf, size, MPI_CHAR, 0, 2, world.mpiComm());
      }
    }
#endif

    iter++;
  }

  // pthread_barrier_wait(&recv_barrier);
  free_memory(s_buf, r_buf, myid);

  sleep(1);

  return 0;
}
