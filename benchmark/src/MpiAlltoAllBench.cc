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

#include <fusion/AlltoAll.h>
#include <fusion/SharedMemory.h>
#include <fusion/rtlx.h>

#include <Debug.h>
#include <Random.h>
#include <timer/Timer.h>
#include <timer/Trace.h>

#include <MPISynchronizedBarrier.h>
#include <MpiAlltoAllBench.h>
#include <Version.h>
#include <parallel/algorithm>

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;
constexpr size_t GB = 1 << 30;

#ifdef NDEBUG
constexpr int nwarmup = 1;
constexpr int niters  = 10;
#else
constexpr int nwarmup = 0;
constexpr int niters  = 1;
#endif

constexpr size_t minblocksize = (1 << 7);
// constexpr size_t minblocksize = 32768 * 2;
/* If maxblocksiz == 0, this means that we use the capacity per node and scale
 * the minblocksize in successive steps */
constexpr size_t maxblocksize = 0;
/* constexpr size_t maxblocksize = runtime argument */

// This are approximately 25 GB
// constexpr size_t capacity_per_node = 32 * MB * 28 * 28;
constexpr size_t capacity_per_node = 16 * GB;

// The container where we store our
using value_t = int;
// using container_t = std::unique_ptr<value_t[]>;
using container_t = mpi::ShmSegment<value_t>;
using iterator_t  = typename container_t::pointer;

using twoSidedA2A_t = std::function<void(
    iterator_t,
    iterator_t,
    int,
    mpi::MpiCommCtx const&,
    merge_t<iterator_t, iterator_t>)>;

std::array<std::pair<std::string, twoSidedA2A_t>, 9> TWO_SIDED = {
    std::make_pair(
        "AlltoAll",
        a2a::MpiAlltoAll<
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseFlatHandshake",
        a2a::scatteredPairwise<
            a2a::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseOneFactor",
        a2a::scatteredPairwise<
            a2a::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake4",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            4>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake8",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            8>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake16",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            16>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor4",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            4>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor8",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            8>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor16",
        a2a::scatteredPairwiseWaitsome<
            a2a::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            16>)
#if 0
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
#endif
// Bruck Algorithms, first the original one, then a modified version which
// omits the last local rotation step
#if 0
    std::make_pair("Bruck", a2a::bruck<iterator_t, iterator_t, merge_t>),
    std::make_pair(
        "Bruck_Mod", a2a::bruck_mod<iterator_t, iterator_t, merge_t>)
#endif
};

using oneSidedA2A_t = std::function<void(
    mpi::ShmSegment<value_t> const&,
    mpi::ShmSegment<value_t>&,
    int,
    merge_t<iterator_t, iterator_t>)>;

std::array<std::pair<std::string, oneSidedA2A_t>, 3> ONE_SIDED = {
    std::make_pair(
        "All2AllNaive",
        a2a::all2allNaive<value_t, merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "All2AllMortonZSource",
        a2a::all2allMortonZSource<value_t, merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "All2AllMortonZDest",
        a2a::all2allMortonZDest<value_t, merge_t<iterator_t, iterator_t>>)};

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::MpiCommCtx worldCtx{MPI_COMM_WORLD};

  auto        sharedCommCtx = mpi::splitSharedComm(worldCtx);
  auto const& commCtx       = sharedCommCtx;

  auto me = commCtx.rank();
  auto nr = commCtx.size();

  if (argc < 2) {
    if (me == 0) {
      std::cout << "usage: " << argv[0] << " [number of nodes] <algorithm>\n";
    }
    MPI_Finalize();
    return 1;
  }

  auto nhosts = std::atoi(argv[1]);

  std::vector<std::pair<std::string, twoSidedA2A_t>> twoSidedAlgos;

  if (argc == 3) {
    std::string selected_algo = argv[2];

    auto algo = std::find_if(
        std::begin(TWO_SIDED),
        std::end(TWO_SIDED),
        [selected_algo](auto const& p) { return p.first == selected_algo; });

    if (algo == std::end(TWO_SIDED)) {
      if (me == 0) {
        std::cout << "invalid algorithm\n";
      }

      MPI_Finalize();
      return 1;
    }

    twoSidedAlgos.push_back(*algo);
  }
  else {
    std::copy(
        std::begin(TWO_SIDED),
        std::end(TWO_SIDED),
        std::back_inserter(twoSidedAlgos));
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
  const size_t maxprocsize = minblocksize;
#endif

  // We have to divide the maximum capacity per proc by the number of PE
  // to get the largest possible block size
  const size_t _maxblocksize =
      (maxblocksize == 0) ? maxprocsize / nr : maxblocksize;

  std::size_t nsteps = std::ceil(std::log2(_maxblocksize)) -
                       std::ceil(std::log2(minblocksize));

  if (nsteps <= 0) {
    nsteps = 1;
  }

  nsteps = std::min<std::size_t>(nsteps, 20);

  if (me == 0) {
    std::cout << "++ number of blocksize steps: " << nsteps << "\n";
    std::cout << "++ minblocksize: " << minblocksize << "\n";
    std::cout << "++ maxlocksize: " << _maxblocksize << "\n";
  }

  P(me << " nsteps: " << nsteps);

  A2A_ASSERT(minblocksize >= sizeof(value_t));

  if (me == 0) {
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(commCtx.mpiComm());
  A2A_ASSERT(is_clock_synced);

  for (size_t blocksize = minblocksize, step = 0; step <= nsteps;
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

    // container_t data, out, correct;
    auto data = container_t(commCtx, nels);
    auto out  = container_t(commCtx, nels);
#ifndef NDEBUG
    auto correct = container_t(commCtx, nels);
#endif

    for (int it = 0; it < niters + nwarmup; ++it) {
#pragma omp parallel
      {
        std::mt19937_64 generator(random_seed_seq::get_instance());
        std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
        for (std::size_t block = 0; block < std::size_t(nr); ++block) {
#ifdef NDEBUG
          // generate some randome values
          std::generate(
              std::next(data.base(), block * sendcount),
              std::next(data.base(), (block + 1) * sendcount),
              [&]() { return distribution(generator); });
          // sort it
          std::sort(
              std::next(data.base(), block * sendcount),
              std::next(data.base(), (block + 1) * sendcount));
#else
          std::iota(
              std::next(data.base(), block * sendcount),
              std::next(data.base(), (block + 1) * sendcount),
              block * sendcount + (me * nels));
#endif
        }
      }

      for (std::size_t block = 0; block < std::size_t(nr); ++block) {
        A2A_ASSERT(std::is_sorted(
            std::next(data.base(), block * sendcount),
            std::next(data.base(), (block + 1) * sendcount)));
      }

      auto merger = [nels](
                        std::vector<std::pair<iterator_t, iterator_t>> seqs,
                        iterator_t                                     res,
                        std::uint16_t nthreads = 0) {
        // parallel merge does not support inplace merging
        // nels must be the number of elements in all sequences
        __gnu_parallel::multiway_merge(
            std::begin(seqs),
            std::end(seqs),
            res,
            nels,
            std::less<value_t>{},
            __gnu_parallel::parallel_tag{nthreads});
      };

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      a2a::MpiAlltoAll(
          data.base(),
          correct.base(),
          static_cast<int>(sendcount),
          commCtx.mpiComm(),
          merger);
      A2A_ASSERT(
          std::is_sorted(correct.base(), std::next(correct.base(), nels)));
#endif

      Params p{};
      p.nhosts    = nhosts;
      p.nprocs    = nr;
      p.me        = me;
      p.step      = step + 1;
      p.nbytes    = nels * nr * sizeof(value_t);
      p.blocksize = blocksize;

      for (auto const& algo : twoSidedAlgos) {
        P("running algorithm: " << algo.first);

        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier

        auto isHypercube = algo.first.find("Hypercube") != std::string::npos;

        if (isHypercube && !a2a::isPow2(static_cast<unsigned>(nr))) {
          continue;
        }

        auto barrier = clock.Barrier(commCtx.mpiComm());
        A2A_ASSERT(barrier.Success(commCtx.mpiComm()));

        auto t = run_algorithm(
            algo.second,
            data.base(),
            out.base(),
            static_cast<int>(sendcount),
            commCtx,
            merger);

#ifndef NDEBUG
        A2A_ASSERT(std::equal(
            correct.base(), std::next(correct.base(), nels), out.base()));
#endif

        if (it >= nwarmup) {
          auto trace = a2a::TimeTrace{me, algo.first};

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

      for (auto const& algo : ONE_SIDED) {
        P("running algorithm: " << algo.first);

        auto barrier = clock.Barrier(commCtx.mpiComm());
        A2A_ASSERT(barrier.Success(commCtx.mpiComm()));

        auto t = ChronoClockNow();
        algo.second(data, out, static_cast<int>(sendcount), merger);
        t = ChronoClockNow() - t;

#ifndef NDEBUG
        A2A_ASSERT(std::equal(
            correct.base(), std::next(correct.base(), nels), out.base()));
#endif

        if (it >= nwarmup) {
          auto trace = a2a::TimeTrace{me, algo.first};

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
    // synchronize before advancing to the next stage
    P(me << " reaching barrier, going to next iteration");
    MPI_Barrier(commCtx.mpiComm());
  }

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
    const std::string&                 algorithm,
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
    auto        flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if ((std::strstr(flag_name.c_str(), "OMPI_") != nullptr) ||
        (std::strstr(flag_name.c_str(), "I_MPI_") != nullptr)) {
      std::cout << "-- " << flag_name << " = " << flag_value << "\n";
    }

    env_var_kv = *(environ + i);
  }
}