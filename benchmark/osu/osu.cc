#include <unistd.h>

#include <cstring>
#include <fmpi/Pinning.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <iostream>
#include <sstream>
#include <tlx/cmdline_parser.hpp>

#include "osu.hpp"

struct Params options;

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

  // cp.add_flag('V', "vary-window", options.window_varied, "Variable
  // Windows");

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

void free_memory_pt2pt_mul(void* sbuf, void* rbuf, int rank, int pairs) {
  free(sbuf);
  free(rbuf);
}
