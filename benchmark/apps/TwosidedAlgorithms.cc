#include <omp.h>

#include <MPISynchronizedBarrier.hpp>
#include <Params.hpp>
#include <TwosidedAlgorithms.hpp>
#include <fmpi/AlltoAll.hpp>
#include <fmpi/Bruck.hpp>
#include <fmpi/Math.hpp>
#include <fmpi/Random.hpp>
#include <parallel/algorithm>
#include <regex>
#include <rtlx/ScopedLambda.hpp>
#include <tlx/algorithm.hpp>
#include <tlx/container/simple_vector.hpp>

#ifdef NDEBUG
constexpr int nwarmup = 1;
#else
constexpr int nwarmup = 0;
#endif

// The container where we store our
using value_t = int;
using storage_t =
    tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;
using iterator_t = typename storage_t::iterator;

template <class InputIterator, class OutputIterator>
using merger_t = std::function<OutputIterator(
    std::vector<std::pair<InputIterator, InputIterator>>, OutputIterator)>;

static std::vector<std::pair<
    std::string,
    fmpi_algrithm_t<iterator_t, merger_t<iterator_t, iterator_t>>>>
    ALGORITHMS = {std::make_pair(
                      "AlltoAll",
                      fmpi::MpiAlltoAll<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "RingWaitall4",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "RingWaitall8",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "RingWaitall16",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "OneFactorWaitall4",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "OneFactorWaitall8",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "OneFactorWaitall16",
                      fmpi::scatteredPairwiseWaitall<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "RingWaitsome4",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "RingWaitsome8",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "RingWaitsome16",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "OneFactorWaitsome4",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "OneFactorWaitsome8",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "OneFactorWaitsome16",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "RingWaitsomeOverlap4",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "RingWaitsomeOverlap8",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "RingWaitsomeOverlap16",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::FlatHandshake,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "OneFactorWaitsomeOverlap4",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "OneFactorWaitsomeOverlap8",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "OneFactorWaitsomeOverlap16",
                      fmpi::scatteredPairwiseWaitsomeOverlap<
                          fmpi::OneFactor,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  // Bruck Algorithms, first the original one, then a modified
                  // version which omits the last local rotation step
                  std::make_pair(
                      "Bruck",
                      fmpi::bruck<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "Bruck_indexed",
                      fmpi::bruck_indexed<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "Bruck_interleave",
                      fmpi::bruck_interleave<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "Bruck_Mod",
                      fmpi::bruck_mod<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>)

};

int main(int argc, char* argv[]) {
  constexpr auto required = MPI_THREAD_SERIALIZED;

  int provided;

  RTLX_ASSERT_RETURNS(
      MPI_Init_thread(&argc, &argv, required, &provided), MPI_SUCCESS);

  auto finalizer = rtlx::scope_exit(
      []() { RTLX_ASSERT_RETURNS(MPI_Finalize(), MPI_SUCCESS); });

  mpi::Context worldCtx{MPI_COMM_WORLD};

  auto me = worldCtx.rank();
  auto nr = worldCtx.size();

  if (provided < required) {
    if (me == 0) {
      std::cout << "MPI_THREAD_SERIALIZED is not supported!\n";
      return 1;
    }
  }

  fmpi::benchmark::Params params{};

  if (!fmpi::benchmark::process(argc, argv, worldCtx, params)) {
    return 1;
  }

  if (!params.pattern.empty()) {
    // remove algorithms not matching a pattern
    ALGORITHMS.erase(
        std::remove_if(
            std::begin(ALGORITHMS),
            std::end(ALGORITHMS),
            [regex = std::regex(params.pattern)](auto const& algo) {
              return !std::regex_match(algo.first, regex);
            }),
        ALGORITHMS.end());
  }

  if (!fmpi::isPow2(nr)) {
    // remove bruck_mod if we have not a power of 2 ranks.
    auto it = std::find_if(
        std::begin(ALGORITHMS), std::end(ALGORITHMS), [](auto const& algo) {
          return algo.first == "Bruck_Mod";
        });

    if (it != std::end(ALGORITHMS)) {
      ALGORITHMS.erase(it);
    }
  }

  RTLX_ASSERT((nr % params.nhosts) == 0);

  if (me == 0) {
    std::cout << "Node Topology:\n";
  }

  MPI_Barrier(worldCtx.mpiComm());

  int32_t const ppn = nr / params.nhosts;

  if (me < ppn) {
    std::ostringstream os;
    os << "  MPI Rank " << me << "\n";
    fmpi::print_config(os);
    std::cout << os.str();
  }

  MPI_Barrier(worldCtx.mpiComm());

  if (me == 0) {
    std::cout << "\n";
    fmpi::benchmark::printBenchmarkPreamble(std::cout, "++ ", "\n");
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(worldCtx.mpiComm());
  RTLX_ASSERT(is_clock_synced);

  FMPI_DBG(params.niters);

  for (std::size_t step = 0; step < params.sizes.size(); ++step) {
    // each process sends sencount to all other PEs
    auto const blocksize = params.sizes[step];
    RTLX_ASSERT(blocksize >= sizeof(value_t));
    RTLX_ASSERT(blocksize % sizeof(value_t) == 0);

    auto sendcount = blocksize / sizeof(value_t);

    FMPI_DBG(sendcount);

    RTLX_ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    RTLX_ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    auto data    = storage_t(nels);
    auto out     = storage_t(nels);
    auto correct = storage_t(0);

    for (auto it = 0; it < static_cast<int>(params.niters) + nwarmup; ++it) {
#pragma omp parallel
      {
        std::mt19937_64 generator(random_seed_seq::get_instance());
        std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
        for (std::size_t block = 0; block < std::size_t(nr); ++block) {
#ifdef NDEBUG
          // generate some random values
          std::generate(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount),
              [&]() { return distribution(generator); });
          // sort it
          std::sort(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount));
#else
          std::iota(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount),
              block * sendcount + (me * nels));
#endif
        }
      }

      for (std::size_t block = 0; block < std::size_t(nr); ++block) {
        RTLX_ASSERT(std::is_sorted(
            std::next(data.begin(), block * sendcount),
            std::next(data.begin(), (block + 1) * sendcount)));
      }

      auto merger = [](std::vector<std::pair<iterator_t, iterator_t>> seqs,
                       iterator_t                                     res) {
        // parallel merge does not support inplace merging
        // nels must be the number of elements in all sequences
        RTLX_ASSERT(seqs.size());
        RTLX_ASSERT(res);

        auto const size = std::accumulate(
            std::begin(seqs),
            std::end(seqs),
            std::size_t(0),
            [](auto acc, auto c) {
              return acc + std::distance(c.first, c.second);
            });

        return tlx::parallel_multiway_merge(
            std::begin(seqs),
            std::end(seqs),
            res,
            size,
            std::less<>{},
            tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
            tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
            omp_get_max_threads());
      };

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      if (params.check) {
        correct = storage_t(nels);
        fmpi::MpiAlltoAll(
            data.begin(),
            correct.begin(),
            static_cast<int>(sendcount),
            worldCtx,
            merger);
      }

      Measurement m{};
      m.nhosts    = params.nhosts;
      m.nprocs    = nr;
      m.nthreads  = omp_get_max_threads();
      m.me        = me;
      m.step      = step + 1;
      m.nbytes    = nels * nr * sizeof(value_t);
      m.blocksize = blocksize;

      for (auto const& algo : ALGORITHMS) {
        FMPI_DBG_STREAM("running algorithm: " << algo.first);

        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(worldCtx.mpiComm());
        RTLX_ASSERT(barrier.Success(worldCtx.mpiComm()));

        auto total = run_algorithm(
            algo.second,
            data.begin(),
            out.begin(),
            static_cast<int>(sendcount),
            worldCtx,
            merger);

        if (params.check) {
          auto check = std::equal(
              correct.begin(), std::next(correct.begin(), nels), out.begin());

          if (!check) {
            std::ostringstream os;
            os << "[ERROR] [Rank " << me << "] " << algo.first
               << ": incorrect sequence (";
            std::copy(
                out.begin(),
                out.end(),
                std::ostream_iterator<value_t>(os, ", "));
            os << ") vs. (";
            std::copy(
                correct.begin(),
                correct.end(),
                std::ostream_iterator<value_t>(os, ", "));
            os << ")\n";
            std::cerr << os.str();
          }
        }

        auto& traceStore = rtlx::TraceStore::GetInstance();

        if (it >= nwarmup) {
          m.algorithm = algo.first;
          m.iter      = it - nwarmup + 1;

          auto traces = traceStore.traces(algo.first);

          // insert total time
          traces.insert(std::make_pair(fmpi::TOTAL, total));

          printMeasurementCsvLine(std::cout, m, traces);
        }

        traceStore.erase(algo.first);
      }

      // synchronize before advancing to the next stage
      FMPI_DBG("Iteration Finished");
      MPI_Barrier(worldCtx.mpiComm());
    }
  }

  return 0;
}

template <class Rep, class Period>
std::ostream& operator<<(
    std::ostream& os, const std::chrono::duration<Rep, Period>& d) {
  os << rtlx::to_seconds(d);
  return os;
}

void printMeasurementHeader(std::ostream& os) {
  os << "Nodes, Procs, Threads, Round, NBytes, Blocksize, Algo, Rank, "
        "Iteration, "
        "Measurement, "
        "Value\n";
}

std::ostream& operator<<(
    std::ostream& os, typename rtlx::TraceStore::value_t const& v) {
  std::visit([&os](auto const& val) { os << val; }, v);
  return os;
}

void printMeasurementCsvLine(
    std::ostream&      os,
    Measurement const& params,
    std::unordered_map<std::string, typename rtlx::TraceStore::value_t> const&
        traces) {
  for (auto&& trace : traces) {
    std::ostringstream myos;
    myos << params.nhosts << ", ";
    myos << params.nprocs << ", ";
    myos << params.nthreads << ", ";
    myos << params.step << ", ";
    myos << params.nbytes << ", ";
    myos << params.blocksize << ", ";
    myos << params.algorithm << ", ";
    myos << params.me << ", ";
    myos << params.iter << ", ";
    myos << trace.first << ", ";
    myos << trace.second << "\n";

    os << myos.str();
  }
}
