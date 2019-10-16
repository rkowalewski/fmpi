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

#include <fmpi/AlltoAll.h>
#include <fmpi/Debug.h>
#include <fmpi/SharedMemory.h>

#include <Random.h>
#include <rtlx/Assert.h>
#include <rtlx/ScopedLambda.h>
#include <rtlx/Timer.h>
#include <rtlx/Trace.h>

#include <MPISynchronizedBarrier.h>
#include <MpiAlltoAllBench.h>
#include <Version.h>
#include <parallel/algorithm>

#include <tlx/cmdline_parser.hpp>

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

using twoSidedAlgo_t = std::function<void(
    iterator_t,
    iterator_t,
    int,
    mpi::MpiCommCtx const&,
    merge_t<iterator_t, iterator_t>)>;

std::array<std::pair<std::string, twoSidedAlgo_t>, 9> TWO_SIDED = {
    std::make_pair(
        "AlltoAll",
        fmpi::MpiAlltoAll<
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseFlatHandshake",
        fmpi::scatteredPairwise<
            fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseOneFactor",
        fmpi::scatteredPairwise<
            fmpi::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake4",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            4>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake8",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            8>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeFlatHandshake16",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            16>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor4",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            4>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor8",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            8>),
    std::make_pair(
        "ScatteredPairwiseWaitsomeOneFactor16",
        fmpi::scatteredPairwiseWaitsome<
            fmpi::AllToAllAlgorithm::ONE_FACTOR,
            iterator_t,
            iterator_t,
            merge_t<iterator_t, iterator_t>,
            16>)
#if 0
    std::make_pair(
        "ScatteredPairwiseWaitany16",
        fmpi::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 16>),
    std::make_pair(
        "ScatteredPairwiseWaitany32",
        fmpi::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 32>),
    std::make_pair(
        "ScatteredPairwiseWaitany64",
        fmpi::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 64>),
    std::make_pair(
        "ScatteredPairwiseWaitany128",
        fmpi::scatteredPairwiseWaitany<iterator_t, iterator_t, merge_t, 128>),
    // Hierarchical XOR Shift Hypercube, works only if #PEs is power of two
    std::make_pair(
        "Hypercube", fmpi::hypercube<iterator_t, iterator_t, merge_t>),
#endif
// Bruck Algorithms, first the original one, then a modified version which
// omits the last local rotation step
#if 0
    std::make_pair("Bruck", fmpi::bruck<iterator_t, iterator_t, merge_t>),
    std::make_pair(
        "Bruck_Mod", fmpi::bruck_mod<iterator_t, iterator_t, merge_t>)
#endif
};

using oneSidedAlgo_t = std::function<void(
    mpi::MpiCommCtx const&,
    mpi::ShmSegment<value_t> const&,
    mpi::ShmSegment<value_t>&,
    int,
    merge_t<iterator_t, iterator_t>)>;

std::array<std::pair<std::string, oneSidedAlgo_t>, 3> ONE_SIDED = {

    std::make_pair(
        "All2AllNaive",
        fmpi::all2allNaive<value_t, merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "All2AllMortonZSource",
        fmpi::all2allMortonZSource<value_t, merge_t<iterator_t, iterator_t>>),
    std::make_pair(
        "All2AllMortonZDest",
        fmpi::all2allMortonZDest<value_t, merge_t<iterator_t, iterator_t>>)};

template <class cT, class traits = std::char_traits<cT>>
class basic_nullbuf : public std::basic_streambuf<cT, traits> {
  typename traits::int_type overflow(typename traits::int_type c)
  {
    return traits::not_eof(c);  // indicate success
  }
};

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::MpiCommCtx worldCtx{MPI_COMM_WORLD};

  auto me = worldCtx.rank();
  auto nr = worldCtx.size();

  tlx::CmdlineParser cp;

  // add description and author
  cp.set_description("Benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  unsigned nhosts = 0;
  cp.add_param_unsigned("nodes", nhosts, "Number of computation nodes");

  std::string selected_algo = "";
  cp.add_opt_param_string(
      "algo", selected_algo, "Select a specific algorithm");

  int good = false;

  // process command line
  if (me == 0) {
    good = cp.process(argc, argv);
  }

  MPI_Bcast(&good, 1, mpi::type_mapper<int>::type(), 0, worldCtx.mpiComm());

  if (!good) {
    return 1;
  }

  MPI_Bcast(&nhosts, 1, mpi::type_mapper<int>::type(), 0, worldCtx.mpiComm());

  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  RTLX_ASSERT((nr % nhosts) == 0);

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

  std::size_t nsteps = 1;

  if (_maxblocksize >= minblocksize) {
    nsteps = std::ceil(std::log2(_maxblocksize)) -
             std::ceil(std::log2(minblocksize));
  }

  nsteps = std::min<std::size_t>(nsteps, 20);

  if (me == 0) {
    std::cout << "++ number of blocksize steps: " << nsteps << "\n";
    std::cout << "++ minblocksize: " << minblocksize << "\n";
    std::cout << "++ maxlocksize: " << _maxblocksize << "\n";
    FMPI_DBG(nsteps);
  }

  RTLX_ASSERT(minblocksize >= sizeof(value_t));

  if (me == 0) {
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(worldCtx.mpiComm());
  RTLX_ASSERT(is_clock_synced);

  for (size_t blocksize = minblocksize, step = 0; step <= nsteps;
       blocksize *= 2, ++step) {
    // each process sends sencount to all other PEs
    auto sendcount = blocksize / sizeof(value_t);

    if (me == 0) {
      FMPI_DBG(sendcount);
    }

    RTLX_ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    RTLX_ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    // container_t data, out, correct;
    auto data = container_t(worldCtx, nels);
    auto out  = container_t(worldCtx, nels);
#ifndef NDEBUG
    auto correct = container_t(worldCtx, nels);
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
              std::next(data.base(worldCtx.rank()), block * sendcount),
              std::next(data.base(worldCtx.rank()), (block + 1) * sendcount),
              [&]() { return distribution(generator); });
          // sort it
          std::sort(
              std::next(data.base(worldCtx.rank()), block * sendcount),
              std::next(data.base(worldCtx.rank()), (block + 1) * sendcount));
#else
          std::iota(
              std::next(data.base(worldCtx.rank()), block * sendcount),
              std::next(data.base(worldCtx.rank()), (block + 1) * sendcount),
              block * sendcount + (me * nels));
#endif
        }
      }

      for (std::size_t block = 0; block < std::size_t(nr); ++block) {
        RTLX_ASSERT(std::is_sorted(
            std::next(data.base(worldCtx.rank()), block * sendcount),
            std::next(data.base(worldCtx.rank()), (block + 1) * sendcount)));
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
            std::less<>{},
            __gnu_parallel::parallel_tag{nthreads});
      };

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      fmpi::MpiAlltoAll(
          data.base(worldCtx.rank()),
          correct.base(worldCtx.rank()),
          static_cast<int>(sendcount),
          worldCtx.mpiComm(),
          merger);
      RTLX_ASSERT(std::is_sorted(
          correct.base(worldCtx.rank()),
          std::next(correct.base(worldCtx.rank()), nels)));
#endif

      Params p{};
      p.nhosts    = nhosts;
      p.nprocs    = nr;
      p.me        = me;
      p.step      = step + 1;
      p.nbytes    = nels * nr * sizeof(value_t);
      p.blocksize = blocksize;

      for (auto const& algo : TWO_SIDED) {
        FMPI_DBG_STREAM("running algorithm: " << algo.first);

        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier

        auto isHypercube = algo.first.find("Hypercube") != std::string::npos;

        if (isHypercube && !fmpi::isPow2(static_cast<unsigned>(nr))) {
          continue;
        }

        auto barrier = clock.Barrier(worldCtx.mpiComm());
        RTLX_ASSERT(barrier.Success(worldCtx.mpiComm()));

        auto t = run_algorithm(
            algo.second,
            data.base(worldCtx.rank()),
            out.base(worldCtx.rank()),
            static_cast<int>(sendcount),
            worldCtx,
            merger);

#ifndef NDEBUG
        RTLX_ASSERT(std::equal(
            correct.base(worldCtx.rank()),
            std::next(correct.base(worldCtx.rank()), nels),
            out.base(worldCtx.rank())));
#endif

        if (it >= nwarmup) {
          auto trace = rtlx::TimeTrace{me, algo.first};

          printMeasurementCsvLine(
              std::cout,
              p,
              algo.first,
              std::make_tuple(
                  t,
                  trace.lookup(fmpi::MERGE),
                  trace.lookup(fmpi::COMMUNICATION)));
          trace.clear();
        }
      }

#if 0
      for (auto const& algo : ONE_SIDED) {
        FMPI_DBG_STREAM("running algorithm: " << algo.first);

        auto barrier = clock.Barrier(commCtx.mpiComm());
        RTLX_ASSERT(barrier.Success(commCtx.mpiComm()));

        auto t = rtlx::ChronoClockNow();
        algo.second(commCtx, data, out, static_cast<int>(sendcount), merger);
        t = rtlx::ChronoClockNow() - t;

#ifndef NDEBUG
        RTLX_ASSERT(std::equal(
            correct.base(commCtx.rank()),
            std::next(correct.base(commCtx.rank()), nels),
            out.base(commCtx.rank())));
#endif

        if (it >= nwarmup) {
          auto trace = rtlx::TimeTrace{me, algo.first};

          printMeasurementCsvLine(
              std::cout,
              p,
              algo.first,
              std::make_tuple(
                  t,
                  trace.lookup(fmpi::MERGE),
                  trace.lookup(fmpi::COMMUNICATION)));
          trace.clear();
        }
      }
#endif
    }
    // synchronize before advancing to the next stage
    FMPI_DBG("Iteration Finished");
    MPI_Barrier(worldCtx.mpiComm());
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
