#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include <AlltoAll.h>
#include <Debug.h>
#include <Random.h>
#include <Timer.h>
#include <Types.h>

#include <MPISynchronizedBarrier.h>
#include <MpiAlltoAllBench.h>

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters = 10;

constexpr size_t minblocksize = KB;
/* constexpr size_t maxblocksize = runtime argument */

// This are approximately 25 GB
// constexpr size_t capacity_per_node = 32 * MB * 28 * 28;
constexpr size_t capacity_per_node = 64 * MB;

// The container where we store our
using value_t     = int;
using container_t = std::unique_ptr<value_t[]>;
using iterator_t  = typename container_t::pointer;

using benchmark_t =
    std::function<void(iterator_t, iterator_t, int, MPI_Comm)>;

std::array<std::pair<std::string, benchmark_t>, 7> algos = {
    // Classic MPI All to All
    std::make_pair("AlltoAll", alltoall::MpiAlltoAll<iterator_t, iterator_t>),
    // One Factorizations based on Graph Theory
    std::make_pair(
        "FactorParty", alltoall::factorParty<iterator_t, iterator_t>),
    std::make_pair(
        "FlatFactor", alltoall::flatFactor<iterator_t, iterator_t>),
    // A Simple Flat Handshake which sends and receives never to/from the same
    // rank
    std::make_pair(
        "FlatHandshake", alltoall::flatHandshake<iterator_t, iterator_t>),
    // Hierarchical XOR Shift Hypercube, works only if #PEs is power of two
    std::make_pair("Hypercube", alltoall::hypercube<iterator_t, iterator_t>),
    // Bruck Algorithms, first the original one, then a modified version which
    // omits the last local rotation step
    std::make_pair("Bruck", alltoall::bruck<iterator_t, iterator_t>),
    std::make_pair("Bruck_Mod", alltoall::bruck_mod<iterator_t, iterator_t>)};

int main(int argc, char* argv[])
{
  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  int         me, nr;
  container_t data, out, correct;

  measurements_t measurements;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  if (argc == 1) {
    if (me == 0) {
      std::cout << "usage: " << argv[0] << " [number of nodes]\n";
    }
    MPI_Finalize();
    return 1;
  }

  ASSERT(nr >= 1);

  auto nhosts = std::atoi(argv[1]);

  if (me == 0) {
    print_env();
  }

  std::mt19937_64 generator(random_seed_seq::get_instance());

  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  ASSERT((nr % nhosts) == 0);

  auto procs_per_node = nr / nhosts;

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  ASSERT(is_clock_synced);

  // We divide by two because we have in and out buffers
  // Then we divide by the number of PEs per node
  // Then we divide again divide by number of PEs to obtain the largest
  // blocksize.
  const size_t maxprocsize = capacity_per_node / (2 * procs_per_node);

  // We have to divide the maximum capacity per proc by the number of PE
  // to get the largest possible block size
  const size_t maxblocksize = maxprocsize / nr;

  auto nsteps = static_cast<size_t>(std::log2(maxblocksize)) -
                static_cast<size_t>(std::log2(minblocksize));

  nsteps = std::min<std::size_t>(nsteps, 15);

  ASSERT(minblocksize >= sizeof(value_t));

  for (size_t blocksize = minblocksize, step = 1; step <= nsteps;
       blocksize *= 2, ++step) {
    // each process sends sencount to all other PEs
    auto sendcount = blocksize / sizeof(value_t);

    ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    data.reset(new value_t[nels]);
    out.reset(new value_t[nels]);
#ifndef NDEBUG
    correct.reset(new value_t[nels]);
#endif

    std::iota(&(data[0]), &(data[nels]), me * nr);

    for (size_t it = 0; it < niters; ++it) {
#ifdef NDEBUG
      std::shuffle(&(data[0]), &(data[nels]), generator);
#endif

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      alltoall::MpiAlltoAll(&(data[0]), &(correct[0]), sendcount, comm);
#endif

      for (auto const& algo : algos) {
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(comm);
        ASSERT(barrier.Success(comm));

        auto t = run_algorithm(
            algo.second, &(data[0]), &(out[0]), sendcount, comm);

        measurements[algo.first].emplace_back(t);

        ASSERT(std::equal(&(correct[0]), &(correct[nels]), &(out[0])));
      }
    }

    std::vector<StringDoublePair> ranking;

    constexpr int root = 0;

    for (auto const& algo : algos) {
      auto mid = (niters / 2);

      auto& results = measurements[algo.first];

      // local median
      std::nth_element(results.begin(), results.begin() + mid, results.end());

      // global median
      auto med = medianReduce(results[mid], root, comm);

      // collect the global median into a vector
      if (me == root) {
        ranking.emplace_back(StringDoublePair{algo.first, med});
      }
    }

    if (me == root) {
      ASSERT(ranking.size() == algos.size());

      if (step == 1) {
        // print header
        std::cout
            << "Nodes, Procs, Round, NBytes.KB, Blocksize, Algo, Time\n";
      }

      // sort the median vector
      std::sort(ranking.begin(), ranking.end());

      for (auto const& m : ranking) {
        std::cout << nhosts << ", " << nr << ", " << step << ", "
                  << nels * nr * sizeof(value_t) / KB << ", " << blocksize
                  << ", " << m.first << ", " << m.second << std::endl;
      }
    }

    // reset measurements for next iteration
    measurements.clear();
  }

  MPI_Finalize();

  return 0;
}

extern char** environ;

void print_env()
{
  int   i          = 1;
  char* env_var_kv = *environ;

  for (; env_var_kv != 0; ++i) {
    // Split into key and value:
    char*       flag_name_cstr  = env_var_kv;
    char*       flag_value_cstr = std::strstr(env_var_kv, "=");
    int         flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if (std::strstr(flag_name.c_str(), "OMPI_") ||
        std::strstr(flag_name.c_str(), "I_MPI_")) {
      std::cout << flag_name << " = " << flag_value << "\n";
    }

    env_var_kv = *(environ + i);
  }
}

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs)
{
  return lhs.second < rhs.second;
}

std::ostream& operator<<(std::ostream& os, StringDoublePair const& p)
{
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}
