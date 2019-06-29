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
#include <omp.h>

#include <AlltoAll.h>
#include <Debug.h>
#include <Random.h>
#include <Timer.h>
#include <Types.h>

#include <MPISynchronizedBarrier.h>
#include <MpiAlltoAllBench.h>
#include <Version.h>
#include <parallel/algorithm>

#define W(X) #X << "=" << X << ", "

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;
constexpr size_t GB = 1 << 30;

#ifdef NDEBUG
constexpr size_t nwarmup = 1;
constexpr size_t niters  = 10;
#else
constexpr size_t nwarmup = 0;
constexpr size_t niters  = 1;
#endif

constexpr size_t minblocksize = 128;
/* constexpr size_t maxblocksize = runtime argument */

// This are approximately 25 GB
// constexpr size_t capacity_per_node = 32 * MB * 28 * 28;
// constexpr size_t capacity_per_node = 16 * GB;
constexpr size_t capacity_per_node = 512 * KB;

// The container where we store our
using value_t     = int;
using container_t = std::unique_ptr<value_t[]>;
using iterator_t  = typename container_t::pointer;

using benchmark_t =
    std::function<void(iterator_t, iterator_t, int, MPI_Comm, merge_t)>;

std::array<std::pair<std::string, benchmark_t>, 14> algos = {
    // Classic MPI All to All
    std::make_pair(
        "AlltoAll", a2a::MpiAlltoAll<iterator_t, iterator_t, merge_t>),
    // One Factorizations based on Graph Theory
    std::make_pair(
        "OneFactor", a2a::oneFactor<iterator_t, iterator_t, merge_t>),
    // A Simple Flat Handshake which sends and receives never to/from the same
    // rank
    std::make_pair(
        "FlatHandshake", a2a::flatHandshake<iterator_t, iterator_t, merge_t>),
    // A scatterd handshake
    std::make_pair(
        "ScatteredPairwise8",
        a2a::scatteredPairwise<iterator_t, iterator_t, merge_t, 8>),
    // A scatterd handshake
    std::make_pair(
        "ScatteredPairwise16",
        a2a::scatteredPairwise<iterator_t, iterator_t, merge_t, 16>),
    std::make_pair(
        "ScatteredPairwise32",
        a2a::scatteredPairwise<iterator_t, iterator_t, merge_t, 32>),
    std::make_pair(
        "ScatteredPairwise64",
        a2a::scatteredPairwise<iterator_t, iterator_t, merge_t, 64>),
    std::make_pair(
        "ScatteredPairwise128",
        a2a::scatteredPairwise<iterator_t, iterator_t, merge_t, 128>),
    std::make_pair(
        "ScatteredPairwiseWaitany8",
        a2a::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 8>),
    std::make_pair(
        "ScatteredPairwiseWaitany16",
        a2a::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 16>),
    std::make_pair(
        "ScatteredPairwiseWaitany32",
        a2a::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 32>),
    std::make_pair(
        "ScatteredPairwiseWaitany64",
        a2a::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 64>),
    std::make_pair(
        "ScatteredPairwiseWaitany128",
        a2a::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 128>),
    // Hierarchical XOR Shift Hypercube, works only if #PEs is power of two
    std::make_pair(
        "Hypercube", a2a::hypercube<iterator_t, iterator_t, merge_t>),
// Bruck Algorithms, first the original one, then a modified version which
// omits the last local rotation step
#if 0
    std::make_pair("Bruck", a2a::bruck<iterator_t, iterator_t, merge_t>),
    std::make_pair(
        "Bruck_Mod", a2a::bruck_mod<iterator_t, iterator_t, merge_t>)
#endif

};

int main(int argc, char* argv[])
{
  int         me, nr;
  container_t data, out, correct;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  if (argc < 2) {
    if (me == 0) {
      std::cout << "usage: " << argv[0] << " [number of nodes] <algorithm>\n";
    }
    MPI_Finalize();
    return 1;
  }

  auto nhosts = std::atoi(argv[1]);

  std::vector<std::pair<std::string, benchmark_t>> selected_algos;

  if (argc == 3) {
    std::string selected_algo = argv[2];

    auto algo = std::find_if(
        std::begin(algos), std::end(algos), [selected_algo](auto const& p) {
          return p.first == selected_algo;
        });

    if (algo == std::end(algos)) {
      if (me == 0) {
        std::cout << "invalid algorithm\n";
      }

      MPI_Finalize();
      return 1;
    }

    selected_algos.push_back(*algo);
  }
  else {
    std::copy(
        std::begin(algos),
        std::end(algos),
        std::back_inserter(selected_algos));
  }

  if (me == 0) {
    print_env();
  }

  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  A2A_ASSERT((nr % nhosts) == 0);

  // We divide by two because we have in and out buffers
  // Then we divide by the number of PEs per node
  // Then we divide again divide by number of PEs to obtain the largest
  // blocksize.
#ifdef NDEBUG
  auto         procs_per_node = nr / nhosts;
  const size_t maxprocsize    = capacity_per_node / (2 * procs_per_node);
#else
  const size_t maxprocsize = minblocksize * nr;
#endif

  // We have to divide the maximum capacity per proc by the number of PE
  // to get the largest possible block size
  const size_t maxblocksize = maxprocsize / nr;

  auto nsteps =
      std::ceil(std::log2(maxblocksize)) - std::ceil(std::log2(minblocksize));

  if (nsteps <= 0) {
    nsteps = 1;
  }

  nsteps = std::min<std::size_t>(nsteps, 20);

  A2A_ASSERT(minblocksize >= sizeof(value_t));

  if (me == 0) {
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  A2A_ASSERT(is_clock_synced);

  for (size_t blocksize = minblocksize, step = 1; step <= nsteps;
       blocksize *= 2, ++step) {
    // each process sends sencount to all other PEs
    auto sendcount = blocksize / sizeof(value_t);

    if (me == 0) {
      P("sendcount: " << sendcount);
    }

    A2A_ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    A2A_ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    data.reset(new value_t[nels]);
    out.reset(new value_t[nels]);
#ifndef NDEBUG
    correct.reset(new value_t[nels]);
#endif

    for (size_t it = 0; it < niters + nwarmup; ++it) {
#pragma omp parallel
      {
        std::mt19937_64 generator(random_seed_seq::get_instance());
        std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
        for (std::size_t block = 0; block < std::size_t(nr); ++block) {
#ifdef NDEBUG
          // generate some randome values
          std::generate(
              &data[block * sendcount],
              &data[(block + 1) * sendcount],
              [&]() { return distribution(generator); });
          // sort it
          std::sort(&data[block * sendcount], &data[(block + 1) * sendcount]);
#else
          std::iota(
              &data[block * sendcount],
              &data[(block + 1) * sendcount],
              block * sendcount + (me * nels));
#endif
        }
      }

      for (std::size_t block = 0; block < std::size_t(nr); ++block) {
        A2A_ASSERT(std::is_sorted(
            &data[block * sendcount], &data[(block + 1) * sendcount]));
      }

      auto merger =
          [&](void* begin1, void* end1, void* begin2, void* end2, void* res) {
#if 0
        std::array<std::pair<value_t*, value_t*>, 2> seqs{
            std::make_pair(
                static_cast<value_t*>(begin1), static_cast<value_t*>(end1)),
            std::make_pair(
                static_cast<value_t*>(begin2), static_cast<value_t*>(end2)),
        };

        __gnu_parallel::multiway_merge(
            std::begin(seqs),
            std::end(seqs),
            static_cast<value_t*>(res),
            nels,
            std::less<value_t>{},
            __gnu_parallel::sequential_tag{});
#else
            std::copy(
                static_cast<value_t*>(begin1),
                static_cast<value_t*>(end1),
                static_cast<value_t*>(res));
            std::copy(
                static_cast<value_t*>(begin2),
                static_cast<value_t*>(end2),
                static_cast<value_t*>(res) +
                    std::distance(
                        static_cast<value_t*>(begin1),
                        static_cast<value_t*>(end1)));

#endif
          };

      auto barrier = clock.Barrier(comm);
      A2A_ASSERT(barrier.Success(comm));

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      a2a::MpiAlltoAll(&(data[0]), &(correct[0]), sendcount, comm, merger);
      A2A_ASSERT(std::is_sorted(&correct[0], &(correct[nels])));
#endif

      Params p;
      p.nhosts    = nhosts;
      p.nprocs    = nr;
      p.me        = me;
      p.step      = step;
      p.nbytes    = nels * nr * sizeof(value_t);
      p.blocksize = blocksize;

      for (auto const& algo : selected_algos) {
        P("running algorithm: " << algo.first);
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier

        auto isHypercube =
            algo.first.find("Hypercube") != std::string::npos;

        if (isHypercube && !a2a::isPow2(static_cast<unsigned>(nr))) {
          continue;
        }

        auto barrier = clock.Barrier(comm);
        A2A_ASSERT(barrier.Success(comm));

        auto t = run_algorithm(
            algo.second, &(data[0]), &(out[0]), sendcount, comm, merger);

#ifndef NDEBUG
        std::sort(&(out[0]), &(out[nels]));
#endif
        A2A_ASSERT(std::equal(&(correct[0]), &(correct[nels]), &(out[0])));
        // measurements[algo.first].emplace_back(t);
        if ((nwarmup > 0 && it > nwarmup) || (nwarmup == 0)) {
          auto trace = a2a::TimeTrace{me, algo.first};

#if 0
          if (!(((nr & (nr - 1)) == 0))) {
            if (algo.first.find("Hypercube") == std::string::npos) {
              if (trace.enabled()) {
                A2A_ASSERT(trace.measurements().size() > 0);
              }
            }
          }
#endif
          printMeasurementCsvLine(
              std::cout,
              p,
              algo.first,
              std::make_tuple(
                  t,
                  trace.lookup(a2a::MERGE),
                  trace.lookup(a2a::COMMUNICATION)));
          trace.clear();
        }
      }
    }
  }

  MPI_Finalize();

  return 0;
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

void printMeasurementHeader(std::ostream& os)
{
  os << "Nodes, Procs, Round, NBytes, Blocksize, Algo, Rank, Ttotal, Tmerge, "
        "Tcomm\n";
}

void printMeasurementCsvLine(
    std::ostream&                      os,
    Params                             m,
    std::string                        algorithm,
    std::tuple<double, double, double> times)
{
  double total, tmerge, tcomm;
  std::tie(total, tmerge, tcomm) = times;
  std::ostringstream myos;
  myos << m.nhosts << ", ";
  myos << m.nprocs << ", ";
  myos << m.step << ", ";
  myos << m.nbytes << ", ";
  myos << m.blocksize << ", ";
  myos << algorithm << ", ";
  myos << m.me << ", ";
  myos << total << ", ";
  myos << tmerge << ", ";
  myos << tcomm << "\n";
  os << myos.str();
}

extern char** environ;

void print_env()
{
  int   i          = 1;
  char* env_var_kv = *environ;

  std::cout << "-- A2A_GIT_COMMIT = " << A2A_GIT_COMMIT << "\n";
  for (; env_var_kv != nullptr; ++i) {
    // Split into key and value:
    char*       flag_name_cstr  = env_var_kv;
    char*       flag_value_cstr = std::strstr(env_var_kv, "=");
    int         flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if ((std::strstr(flag_name.c_str(), "OMPI_") != nullptr) ||
        (std::strstr(flag_name.c_str(), "I_MPI_") != nullptr)) {
      std::cout << "-- " << flag_name << " = " << flag_value << "\n";
    }

    env_var_kv = *(environ + i);
  }
}
