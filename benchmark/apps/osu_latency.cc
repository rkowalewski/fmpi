#include <unistd.h>

#include <cstring>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <iostream>
#include <rtlx/ScopedLambda.hpp>
#include <tlx/cmdline_parser.hpp>

static constexpr std::size_t LARGE_MESSAGE_SIZE = 8192;
static constexpr int         FIELD_WIDTH        = 18;
static constexpr int         FLOAT_PRECISION    = 2;

struct Params {
  constexpr Params()            = default;
  unsigned int iterations       = 10000;
  unsigned int iterations_large = 1000;
  unsigned int warmups          = 100;
  unsigned int warmups_large    = 10;
  std::size_t  smin             = 0;
  std::size_t  smax             = 4096 * 1024;  // 4k
};

bool read_input(int argc, char* argv[]);
int  allocate_memory_pt2pt(char** sbuf, char** rbuf, int rank);
void set_buffer_pt2pt(void* buffer, int rank, int data, size_t size);

char *s_buf, *r_buf;

fmpi::collective_future irecv(
    void*        buf,
    int          count,
    MPI_Datatype datatype,
    int          source,
    int          tag,
    MPI_Comm     comm) {
  auto request = std::make_unique<MPI_Request>();
  MPI_Irecv(buf, count, datatype, source, tag, comm, request.get());
  return fmpi::make_mpi_future(std::move(request));
}

Params params{};

int main(int argc, char* argv[]) {
  // Our value type
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  const auto& world = mpi::Context::world();

  if (world.size() != 2) {
    if (world.rank() == 0) {
      fprintf(stderr, "This test requires exactly two processes\n");
    }

    return EXIT_FAILURE;
  }

  if (!read_input(argc, argv)) {
    return 0;
  }

  auto const myid = world.rank();
  if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
    return EXIT_FAILURE;
  }

  double     t_start, t_end;
  MPI_Status reqstat;
  // MPI_Request req;
  if (myid == 0) {
    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
  }
  /* Latency test */
  for (auto size = params.smin; size <= params.smax;
       size      = (size ? size * 2 : 1)) {
    set_buffer_pt2pt(s_buf, myid, 'a', size);
    set_buffer_pt2pt(r_buf, myid, 'b', size);

    if (size > LARGE_MESSAGE_SIZE) {
      params.iterations = params.iterations_large;
      params.warmups    = params.warmups_large;
    }

    MPI_Barrier(world.mpiComm());

    if (myid == 0) {
      for (uint32_t i = 0; i < params.iterations + params.warmups; i++) {
        if (i == params.warmups) {
          t_start = MPI_Wtime();
        }

        MPI_Send(s_buf, size, MPI_CHAR, 1, 1, world.mpiComm());

#if 0
        MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, world.mpiComm(), &reqstat);
#else
        auto req = std::make_unique<MPI_Request>();
        MPI_Irecv(r_buf, size, MPI_CHAR, 1, 1, world.mpiComm(), req.get());
        auto future = fmpi::make_mpi_future(std::move(req));
#endif
      }

      t_end = MPI_Wtime();
    }

    else if (myid == 1) {
      for (uint32_t i = 0; i < params.iterations + params.warmups; i++) {
        MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, world.mpiComm(), &reqstat);
        MPI_Send(s_buf, size, MPI_CHAR, 0, 1, world.mpiComm());
      }
    }

    if (myid == 0) {
      double latency = (t_end - t_start) * 1e6 / (2.0 * params.iterations);

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
  }
}

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
}  // namespace detail
bool read_input(int argc, char* argv[]) {
  tlx::CmdlineParser cp;
  using onullstream = detail::basic_onullstream<char>;

  // add description and author
  cp.set_description("Latency benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_bytes(
      's', "smin", params.smin, "Minimum message size sent to each peer.");
  cp.add_bytes(
      'S', "smax", params.smax, "Maximum message size sent to each peer.");

  cp.add_uint(
      'i', "iterations_small", params.iterations, "Trials per round.");
  cp.add_uint(
      'I', "iterations_large", params.iterations_large, "Trials per round.");

  cp.add_uint('w', "warmups_small", params.warmups, "Warmups per round.");
  cp.add_uint(
      'W', "warmups_large", params.warmups_large, "Warmups per round.");

  if (mpi::Context::world().rank() == 0) {
    return cp.process(argc, argv, std::cout);
  } else {
    onullstream os;
    return cp.process(argc, argv, os);
  }
}

int allocate_memory_pt2pt(char** sbuf, char** rbuf, int rank) {
  auto const align_size = sysconf(_SC_PAGESIZE);
  if (posix_memalign((void**)sbuf, align_size, params.smax)) {
    fprintf(stderr, "Error allocating host memory\n");
    return 1;
  }

  if (posix_memalign((void**)rbuf, align_size, params.smax)) {
    fprintf(stderr, "Error allocating host memory\n");
    return 1;
  }
  return 0;
}

void set_buffer_pt2pt(void* buffer, int /*rank*/, int data, size_t size) {
  std::memset(buffer, data, size);
}
