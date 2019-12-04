#include <MPISynchronizedBarrier.h>
#include <MpiAlltoAllBench.h>
#include <Params.h>
#include <Random.h>
#include <Version.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstring>
#include <fmpi/AlltoAll.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/SharedMemory.hpp>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <parallel/algorithm>
#include <rtlx/Assert.hpp>
#include <rtlx/ScopedLambda.hpp>
#include <rtlx/Timer.hpp>
#include <rtlx/Trace.hpp>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

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

// The container where we store our
using value_t = int;
// using storage_t = std::unique_ptr<value_t[]>;
using storage_t  = mpi::ShmSegment<value_t>;
using iterator_t = typename storage_t::pointer;

using twoSidedAlgo_t = std::function<void(
    iterator_t,
    iterator_t,
    int,
    mpi::Context const&,
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
    mpi::Context const&,
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

  mpi::Context worldCtx{MPI_COMM_WORLD};

  auto me = worldCtx.rank();
  auto nr = worldCtx.size();

  fmpi::benchmark::Params params{};

  if (!fmpi::benchmark::process(argc, argv, worldCtx, params)) {
    return -1;
  }

  RTLX_ASSERT((nr % params.nhosts) == 0);

  std::size_t nsteps = std::ceil(std::log2(params.maxblocksize)) -
                       std::ceil(std::log2(params.minblocksize));

  nsteps = std::min(nsteps, std::size_t(20));

  RTLX_ASSERT(params.minblocksize >= sizeof(value_t));

  if (me == 0) {
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(worldCtx.mpiComm());
  RTLX_ASSERT(is_clock_synced);

  FMPI_DBG(niters);

  for (size_t blocksize = params.minblocksize, step = 0; step <= nsteps;
       blocksize *= 2, ++step) {
    // each process sends sencount to all other PEs
    auto sendcount = blocksize / sizeof(value_t);

    FMPI_DBG(sendcount);

    RTLX_ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    RTLX_ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    // storage_t data, out, correct;
    auto data = storage_t(worldCtx, nels);
    auto out  = storage_t(worldCtx, nels);
#ifndef NDEBUG
    auto correct = storage_t(worldCtx, nels);
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
      p.nhosts    = params.nhosts;
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
