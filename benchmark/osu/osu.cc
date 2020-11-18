#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fmpi/Pinning.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>
#include <iostream>
#include <sstream>
#include <tlx/cmdline_parser.hpp>

#include "osu.hpp"

struct Params options;

/* A is the A in DAXPY for the Compute Kernel */
#define A 2.0
#define DEBUG 0
/*
 * We are using a 2-D matrix to perform dummy
 * computation in non-blocking collective benchmarks
 */
#define DIM 25
static float **a, *x, *y;

namespace detail {
template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf : public std::basic_streambuf<cT, traits> {
  auto overflow(typename traits::int_type c) ->
      typename traits::int_type override {
    return traits::not_eof(c);  // indicate success
  }
};

template <class cT, class traits = std::char_traits<cT> >
class basic_onullstream : public std::basic_ostream<cT, traits> {
 public:
  basic_onullstream()
    : std::basic_ios<cT, traits>(&m_sbuf)
    , std::basic_ostream<cT, traits>(&m_sbuf) {
    this->init(&m_sbuf);
  }

 private:
  basic_nullbuf<cT, traits> m_sbuf;
};

static constexpr auto trace_name = std::string_view("schedule_ctx");

static constexpr auto t_complete_all = std::string_view("t_complete_all");
static constexpr auto t_complete_any = std::string_view("t_complete_any");
static constexpr auto t_dispatch     = std::string_view("t_dispatch");
static constexpr auto t_copy         = std::string_view("t_copy");
static constexpr auto t_test_all     = std::string_view("t_test_all");

std::array<std::string_view, 5> trace_names = {
    t_complete_all, t_test_all, t_complete_any, t_dispatch, t_copy};
std::array<std::string_view, 5> trace_headers = {
    "wait_all (us)",
    "test_all (us)",
    "wait_any (us)",
    "dispatch(us)",
    "copy (us)"};

}  // namespace detail

bool read_input(int argc, char* argv[]) {
  tlx::CmdlineParser cp;
  using onullstream = detail::basic_onullstream<char>;

  // add description and author
  cp.set_description("Latency benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_bytes(
      's', "smin", options.smin, "Minimum message size sent to each peer.");
  cp.add_bytes(
      'S', "smax", options.smax, "Maximum message size sent to each peer.");

  cp.add_uint(
      'i', "iterations_small", options.iterations, "Trials per round.");
  cp.add_uint(
      'I', "iterations_large", options.iterations_large, "Trials per round.");

  cp.add_uint('w', "warmups_small", options.warmups, "Warmups per round.");
  cp.add_uint(
      'W', "warmups_large", options.warmups_large, "Warmups per round.");

  cp.add_int('t', "sender_threads", options.sender_threads, "Sender threads");
  cp.add_int(
      'T', "receiver_threads", options.num_threads, "Receiver threads");

  cp.add_int('a', "algorithm", options.algorithm, "Algorithm selection");

  // cp.add_flag('V', "vary-window", options.window_varied, "Variable
  // Windows");

  if (mpi::Context::world().rank() == 0) {
    auto good = cp.process(argc, argv, std::cout);
    if (good) {
      cp.print_result();
    }
    return good;
  } else {
    onullstream os;
    return cp.process(argc, argv, os);
  }
}

int allocate_memory_pt2pt(char** sbuf, char** rbuf, int /*rank*/) {
  auto const align_size = sysconf(_SC_PAGESIZE);
  if (posix_memalign((void**)sbuf, align_size, options.smax)) {
    fprintf(stderr, "Error allocating host memory\n");
    return 1;
  }

  if (posix_memalign((void**)rbuf, align_size, options.smax)) {
    fprintf(stderr, "Error allocating host memory\n");
    return 1;
  }
  return 0;
}

int allocate_memory_pt2pt_mul(char** sbuf, char** rbuf, int rank, int pairs) {
  unsigned long align_size = sysconf(_SC_PAGESIZE);

  if (rank < pairs) {
    if (posix_memalign((void**)sbuf, align_size, options.smax)) {
      fprintf(stderr, "Error allocating host memory\n");
      return 1;
    }

    if (posix_memalign((void**)rbuf, align_size, options.smax)) {
      fprintf(stderr, "Error allocating host memory\n");
      return 1;
    }

    memset(*sbuf, 0, options.smax);
    memset(*rbuf, 0, options.smax);
  } else {
    if (posix_memalign((void**)sbuf, align_size, options.smax)) {
      fprintf(stderr, "Error allocating host memory\n");
      return 1;
    }

    if (posix_memalign((void**)rbuf, align_size, options.smax)) {
      fprintf(stderr, "Error allocating host memory\n");
      return 1;
    }
    memset(*sbuf, 0, options.smax);
    memset(*rbuf, 0, options.smax);
  }

  return 0;
}

void set_buffer_pt2pt(void* buffer, int /*rank*/, int data, size_t size) {
  std::memset(buffer, data, size);
}

uint32_t num_nodes(mpi::Context const& comm) {
  auto const shared_comm = mpi::splitSharedComm(comm);
  int const  is_rank0    = static_cast<int>(shared_comm.rank() == 0);
  int        nhosts      = 0;

  MPI_Allreduce(
      &is_rank0,
      &nhosts,
      1,
      mpi::type_mapper<int>::type(),
      MPI_SUM,
      comm.mpiComm());

  return nhosts;
}

void print_topology(mpi::Context const& ctx, std::size_t nhosts) {
  auto const me   = ctx.rank();
  auto const ppn  = static_cast<int32_t>(ctx.size() / nhosts);
  auto const last = mpi::Rank{ppn - 1};

  mpi::Rank left{};
  mpi::Rank right{};

  if (ppn == 1) {
    left  = (me == 0) ? mpi::Rank::null() : me - 1;
    right = (me == ctx.size() - 1) ? mpi::Rank::null() : me + 1;
  } else {
    left  = (me > 0 && me <= last) ? me - 1 : mpi::Rank::null();
    right = (me < last) ? me + 1 : mpi::Rank::null();
  }

  if (not(left or right)) {
    return;
  }

  char dummy = 0;

  std::ostringstream os;

  if (me == 0) {
    os << "Node Topology:\n";
  }

  MPI_Recv(&dummy, 1, MPI_CHAR, left, 0xAB, ctx.mpiComm(), MPI_STATUS_IGNORE);

  if (me < ppn) {
    os << "  MPI Rank " << me << "\n";
    fmpi::print_pinning(os);
  }

  if (me == last) {
    os << "\n";
  }

  std::cout << os.str() << std::endl;

  MPI_Send(&dummy, 1, MPI_CHAR, right, 0xAB, ctx.mpiComm());

  if (ppn > 1) {
    if (me == 0) {
      MPI_Recv(
          &dummy, 1, MPI_CHAR, last, 0xAB, ctx.mpiComm(), MPI_STATUS_IGNORE);
    } else if (me == last) {
      MPI_Send(&dummy, 1, MPI_CHAR, 0, 0xAB, ctx.mpiComm());
    }
  }
}

void free_memory_pt2pt_mul(
    void* sbuf, void* rbuf, int /*rank*/, int /*pairs*/) {
  free(sbuf);
  free(rbuf);
}

void free_memory(void* sbuf, void* rbuf, int rank) {
  free(sbuf);
  free(rbuf);
}

void allocate_host_arrays() {
  int i = 0, j = 0;
  a = (float**)malloc(DIM * sizeof(float*));

  for (i = 0; i < DIM; i++) {
    a[i] = (float*)malloc(DIM * sizeof(float));
  }

  x = (float*)malloc(DIM * sizeof(float));
  y = (float*)malloc(DIM * sizeof(float));

  for (i = 0; i < DIM; i++) {
    x[i] = y[i] = 1.0f;
    for (j = 0; j < DIM; j++) {
      a[i][j] = 2.0f;
    }
  }
}

int allocate_memory_coll(void** buffer, size_t size) {
  allocate_host_arrays();

  size_t alignment = sysconf(_SC_PAGESIZE);

  return posix_memalign(buffer, alignment, size);
}

void free_host_arrays() {
  int i = 0;

  if (x) {
    free(x);
  }
  if (y) {
    free(y);
  }

  if (a) {
    for (i = 0; i < DIM; i++) {
      free(a[i]);
    }
    free(a);
  }

  x = NULL;
  y = NULL;
  a = NULL;
}

void free_buffer(void* buffer) {
  free(buffer);

  free_host_arrays();
}

void print_preamble_nbc(int rank, std::string_view name) {
  if (rank) {
    return;
  }

  fprintf(stdout, "\n");

  printf("%s\n", name.data());

  fprintf(
      stdout, "# Overall = Coll. Init + Compute + MPI_Test + MPI_Wait\n\n");

  fprintf(stdout, "%-*s", 10, "# Size");
  fprintf(stdout, "%*s", 10, "Window");
  fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");

  fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
  fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
  fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
  fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
  fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
  fprintf(stdout, "%*s", FIELD_WIDTH, "Overlap(%)");

  for (auto&& h : detail::trace_headers) {
    fprintf(stdout, "%*s", FIELD_WIDTH, h.data());
  }

  fprintf(stdout, "\n");

  fflush(stdout);
}

void init_arrays(double target_time) {
#if defined(FMPI_DEBUG_ASSERT) && (FMPI_DEBUG_ASSERT == 1)
  fprintf(
      stderr,
      "called init_arrays with target_time = %f \n",
      (target_time * 1e6));
#endif

#ifdef _ENABLE_CUDA_KERNEL_
  if (options.target == GPU || options.target == BOTH) {
    /* Setting size of arrays for Dummy Compute */
    int N = options.device_array_size;

    /* Device Arrays for Dummy Compute */
    allocate_device_arrays(N);

    double t1 = 0.0, t2 = 0.0;

    while (1) {
      t1 = MPI_Wtime();

      if (options.target == GPU || options.target == BOTH) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        call_kernel(A, d_x, d_y, N, &stream);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaStreamDestroy(stream));
      }

      t2 = MPI_Wtime();
      if ((t2 - t1) < target_time) {
        N += 32;

        /* Now allocate arrays of size N */
        allocate_device_arrays(N);
      } else {
        break;
      }
    }

    /* we reach here with desired N so save it and pass it to options */
    options.device_array_size = N;
    if (DEBUG) {
      fprintf(stderr, "correct N = %d\n", N);
    }
  }
#endif
}

void compute_on_host() {
  int i = 0, j = 0;
  for (i = 0; i < DIM; i++)
    for (j = 0; j < DIM; j++) x[i] = x[i] + a[i][j] * a[j][i] + y[j];
}

static inline void do_compute_cpu(double target_seconds) {
  double t1 = 0.0, t2 = 0.0;
  double time_elapsed = 0.0;
  while (time_elapsed < target_seconds) {
    t1 = MPI_Wtime();
    compute_on_host();
    t2 = MPI_Wtime();
    time_elapsed += (t2 - t1);
  }
  if (DEBUG) {
    fprintf(stderr, "time elapsed = %f\n", (time_elapsed * 1e6));
  }
}

double do_compute_and_probe(double seconds) {
  double     test_time                  = 0.0;
  int        num_tests                  = 0;
  double     target_seconds_for_compute = 0.0;
  int        flag                       = 0;
  MPI_Status status;

  target_seconds_for_compute = seconds;

#ifdef _ENABLE_CUDA_KERNEL_
  double t1 = 0.0;
  double t2 = 0.0;
  if (options.target == GPU) {
    if (options.num_probes) {
      /* Do the dummy compute on GPU only */
      do_compute_gpu(target_seconds_for_compute);
      num_tests = 0;
      while (num_tests < options.num_probes) {
        t1 = MPI_Wtime();
        MPI_CHECK(MPI_Test(request, &flag, &status));
        t2 = MPI_Wtime();
        test_time += (t2 - t1);
        num_tests++;
      }
    } else {
      do_compute_gpu(target_seconds_for_compute);
    }
  } else if (options.target == BOTH) {
    if (options.num_probes) {
      /* Do the dummy compute on GPU and CPU*/
      do_compute_gpu(target_seconds_for_compute);
      num_tests = 0;
      while (num_tests < options.num_probes) {
        t1 = MPI_Wtime();
        MPI_CHECK(MPI_Test(request, &flag, &status));
        t2 = MPI_Wtime();
        test_time += (t2 - t1);
        num_tests++;
        do_compute_cpu(target_seconds_for_compute);
      }
    } else {
      do_compute_gpu(target_seconds_for_compute);
      do_compute_cpu(target_seconds_for_compute);
    }
  } else
#endif
    do_compute_cpu(target_seconds_for_compute);

#ifdef _ENABLE_CUDA_KERNEL_
  if (options.target == GPU || options.target == BOTH) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif

  return test_time;
}

void print_stats_nbc(
    int                              rank,
    int                              size,
    uint32_t                         window_size,
    double                           overall_time,
    double                           cpu_time,
    double                           comm_time,
    double                           wait_time,
    double                           init_time,
    double                           test_time,
    fmpi::FixedVector<double> const& dispatch_times) {
  if (rank) {
    return;
  }

  double overlap;

  /* Note : cpu_time received in this function includes time for
   *      dummy compute as well as test calls so we will subtract
   *      the test_time for overlap calculation as test is an
   *      overhead
   */

  overlap = std::max<double>(
      0, 100 - (((overall_time - (cpu_time - test_time)) / comm_time) * 100));

  fprintf(stdout, "%-*d", 10, size);
  fprintf(stdout, "%*d", 10, window_size);
  fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);

  fprintf(
      stdout,
      "%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f",
      FIELD_WIDTH,
      FLOAT_PRECISION,
      (cpu_time - test_time),
      FIELD_WIDTH,
      FLOAT_PRECISION,
      init_time,
      FIELD_WIDTH,
      FLOAT_PRECISION,
      test_time,
      FIELD_WIDTH,
      FLOAT_PRECISION,
      wait_time,
      FIELD_WIDTH,
      FLOAT_PRECISION,
      comm_time,
      FIELD_WIDTH,
      FLOAT_PRECISION,
      overlap);

  for (auto&& i : fmpi::range(detail::trace_names.size())) {
    fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, dispatch_times[i]);
  }

  fprintf(stdout, "\n");

  fflush(stdout);
}
void calculate_and_print_stats(
    int                 rank,
    int                 size,
    int                 numprocs,
    double              timer,
    double              latency,
    double              test_time,
    double              cpu_time,
    double              wait_time,
    double              init_time,
    mpi::Context const& ctx,
    uint32_t            window_size) {
  double test_total   = (test_time * 1e6) / options.iterations;
  double tcomp_total  = (cpu_time * 1e6) / options.iterations;
  double overall_time = (timer * 1e6) / options.iterations;
  double wait_total   = (wait_time * 1e6) / options.iterations;
  double init_total   = (init_time * 1e6) / options.iterations;
  double comm_time    = latency;

  auto& trace_store = fmpi::TraceStore::instance();

  auto my_times = fmpi::FixedVector<double>(detail::trace_names.size(), 0.0);
  auto my_times_red =
      fmpi::FixedVector<double>(detail::trace_names.size(), 0.0);

  if (not trace_store.empty()) {
    auto const& traces = trace_store.traces(detail::trace_name);

    using double_usecs =
        std::chrono::duration<double, std::chrono::microseconds::period>;

    for (auto&& idx : fmpi::range(detail::trace_names.size())) {
      auto const trace_name = detail::trace_names[idx];

      auto it = traces.find(std::string(trace_name));
      if (it != traces.end()) {
        auto const   t_val = it->second;
        double const us =
            std::chrono::duration_cast<double_usecs>(t_val).count();
        my_times[idx] = us / options.iterations;
      }
    }

    fmpi::TraceStore::instance().erase(detail::trace_name);
  }

  FMPI_ASSERT(fmpi::TraceStore::instance().empty());

  if (rank != 0) {
    MPI_CHECK(MPI_Reduce(
        &test_total, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        &comm_time, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        &overall_time,
        &overall_time,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        &tcomp_total,
        &tcomp_total,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        &wait_total, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        &init_total, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));

    MPI_CHECK(MPI_Reduce(
        my_times.data(),
        my_times.data(),
        static_cast<int>(my_times.size()),
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
  } else {
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE,
        &overall_time,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE,
        &tcomp_total,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0, ctx.mpiComm()));
    MPI_CHECK(MPI_Reduce(
        my_times.data(),
        my_times_red.data(),
        static_cast<int>(my_times.size()),
        MPI_DOUBLE,
        MPI_SUM,
        0,
        ctx.mpiComm()));
  }

  MPI_CHECK(MPI_Barrier(ctx.mpiComm()));

  /* Overall Time (Overlapped) */
  overall_time = overall_time / numprocs;
  /* Computation Time */
  tcomp_total = tcomp_total / numprocs;
  /* Time taken by MPI_Test calls */
  test_total = test_total / numprocs;
  /* Pure Communication Time */
  comm_time = comm_time / numprocs;
  /* Time for MPI_Wait() call */
  wait_total = wait_total / numprocs;
  /* Time for the NBC call */
  init_total = init_total / numprocs;

  for (auto& t : my_times_red) {
    t = t / numprocs;
  }

  print_stats_nbc(
      rank,
      size,
      window_size,
      overall_time,
      tcomp_total,
      comm_time,
      wait_total,
      init_total,
      test_total,
      my_times_red);
}
