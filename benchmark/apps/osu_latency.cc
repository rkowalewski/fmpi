#include <unistd.h>

#include <cstring>
#include <fmpi/Pinning.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <iostream>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
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

auto irecv(
    void*               buf,
    int                 count,
    MPI_Datatype        datatype,
    int                 source,
    int                 tag,
    mpi::Context const& comm,
    MPI_Request&        request) {
  return MPI_Irecv(
      buf, count, datatype, source, tag, comm.mpiComm(), &request);
  // return fmpi::make_mpi_future(std::move(request));
  // return future;
}

auto isend(
    void*               buf,
    int                 count,
    MPI_Datatype        datatype,
    int                 dest,
    int                 tag,
    mpi::Context const& comm,
    MPI_Request&        request) {
  // auto future = fmpi::make_mpi_future();
  return MPI_Isend(buf, count, datatype, dest, tag, comm.mpiComm(), &request);
  // return fmpi::make_mpi_future(std::move(request));
  // return future;
}

void     print_topology(mpi::Context const& ctx, std::size_t nhosts);
uint32_t num_nodes(mpi::Context const& comm);

Params options{};

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

  print_topology(world, num_nodes(world));

  auto const myid = world.rank();
  if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
    return EXIT_FAILURE;
  }

  double     t_start, t_end, t_init;
  MPI_Status reqstat;
  // MPI_Request req;
  if (myid == 0) {
    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
  }

  auto rsend = fmpi::make_mpi_future();
  auto rrecv = fmpi::make_mpi_future();

  /* Latency test */
  for (auto size = options.smin; size <= options.smax;
       size      = (size ? size * 2 : 1)) {
    set_buffer_pt2pt(s_buf, myid, 'a', size);
    set_buffer_pt2pt(r_buf, myid, 'b', size);

    if (size > LARGE_MESSAGE_SIZE) {
      options.iterations = options.iterations_large;
      options.warmups    = options.warmups_large;
    }

    MPI_Barrier(world.mpiComm());

    auto& dispatcher = fmpi::static_dispatcher_pool();

    double t_init_total = 0, t_wait_total = 0, timer = 0;

    if (myid == 0) {
      for (uint32_t i = 0; i < options.iterations + options.warmups; i++) {
        t_start = MPI_Wtime();

#if 1
        auto promise        = fmpi::collective_promise{};
        auto future         = promise.get_future();
        auto schedule_state = std::make_unique<fmpi::ScheduleCtx>(
            std::array<std::size_t, 2>{
                1,
                1,
            },
            std::move(promise));

        // submit into dispatcher
        auto const hdl = dispatcher.submit(std::move(schedule_state));

        auto send = fmpi::make_send(
            s_buf, size, MPI_CHAR, mpi::Rank{1}, 1, world.mpiComm());
        auto recv = fmpi::make_receive(
            r_buf, size, MPI_CHAR, mpi::Rank{1}, 1, world.mpiComm());
        dispatcher.schedule(hdl, fmpi::message_type::ISEND, send);
        dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv);
        dispatcher.commit(hdl);

        t_init = MPI_Wtime();

        future.wait();

#else

        FMPI_ASSERT(rsend.native_handle() == MPI_REQUEST_NULL);
        FMPI_ASSERT(rrecv.native_handle() == MPI_REQUEST_NULL);
        isend(s_buf, size, MPI_CHAR, 1, 1, world, rsend.native_handle());
        irecv(r_buf, size, MPI_CHAR, 1, 1, world, rrecv.native_handle());

        t_init = MPI_Wtime();

        rsend.wait();
        rrecv.wait();
#endif

        t_end = MPI_Wtime();

        if (i >= options.warmups) {
          timer += t_end - t_start;
          t_init_total += t_init - t_start;
          t_wait_total += t_end - t_init;
        }
      }

    }

    else if (myid == 1) {
      for (uint32_t i = 0; i < options.iterations + options.warmups; i++) {
#if 0
        MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, world.mpiComm(), &reqstat);
        MPI_Send(s_buf, size, MPI_CHAR, 0, 1, world.mpiComm());
#else
        FMPI_ASSERT(rsend.native_handle() == MPI_REQUEST_NULL);
        FMPI_ASSERT(rrecv.native_handle() == MPI_REQUEST_NULL);
        irecv(r_buf, size, MPI_CHAR, 0, 1, world, rrecv.native_handle());
        isend(s_buf, size, MPI_CHAR, 0, 1, world, rsend.native_handle());
        rrecv.wait();
        rsend.wait();
        // MPI_Wait(fut_recv.get(), &reqstat);
        // MPI_Wait(fut_send.get(), &reqstat);
#endif
      }
    }

    if (myid == 0) {
      double total   = timer * 1e6 / options.iterations;
      double latency = total / 2;
      double init    = t_init_total * 1e6 / options.iterations;
      double wait    = (t_wait_total)*1e6 / options.iterations;

      fprintf(
          stdout,
          "%-*ld%*.*f%*.*f%*.*f%*.*f\n",
          10,
          size,
          FIELD_WIDTH,
          FLOAT_PRECISION,
          latency,
          FIELD_WIDTH,
          FLOAT_PRECISION,
          total,
          FIELD_WIDTH,
          FLOAT_PRECISION,
          init,
          FIELD_WIDTH,
          FLOAT_PRECISION,
          wait);
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

  if (mpi::Context::world().rank() == 0) {
    return cp.process(argc, argv, std::cout);
  } else {
    onullstream os;
    return cp.process(argc, argv, os);
  }
}

int allocate_memory_pt2pt(char** sbuf, char** rbuf, int rank) {
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

void set_buffer_pt2pt(void* buffer, int /*rank*/, int data, size_t size) {
  std::memset(buffer, data, size);
}

void print_topology(mpi::Context const& ctx, std::size_t nhosts) {
  auto const me   = ctx.rank();
  auto const ppn  = static_cast<int32_t>(ctx.size() / nhosts);
  auto const last = mpi::Rank{ppn - 1};

  auto left  = (me > 0 && me <= last) ? me - 1 : mpi::Rank::null();
  auto right = (me < last) ? me + 1 : mpi::Rank::null();

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

  if (me == 0) {
    MPI_Recv(
        &dummy, 1, MPI_CHAR, last, 0xAB, ctx.mpiComm(), MPI_STATUS_IGNORE);
  } else if (me == last) {
    MPI_Send(&dummy, 1, MPI_CHAR, 0, 0xAB, ctx.mpiComm());
  }
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
