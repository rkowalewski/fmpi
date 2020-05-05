#include <mpi.h>

#include <benchmark/benchmark.h>

#include <thread>
#include <tlx/algorithm.hpp>
#include <vector>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/mpi/Environment.hpp>

using value_t = int;

static constexpr bool debug =
#ifndef NDEBUG
    true
#else
    false
#endif
    ;

struct Params {
  std::size_t nprocs;
  std::size_t nblocks;
  std::size_t blocksz;
  std::size_t windowsz;
  std::size_t arraysize;
};

std::ostream& operator<<(std::ostream& os, Params const& p) {
  std::ostringstream ss;

  ss << "{nprocs: " << p.nprocs << ", ";
  ss << "nblocks: " << p.nblocks << ", ";
  ss << "blocksz: " << p.blocksz << ", ";
  ss << "windowsz: " << p.windowsz << ", ";
  ss << "arraysize: " << p.arraysize << "}\n";

  os << ss.str();

  return os;
}

static Params processParams(benchmark::State const& state) {
  Params params;

  params.nprocs   = state.range(0);
  params.blocksz  = state.range(1);
  params.windowsz = state.range(2);

  params.nblocks = params.nprocs /* * params.nprocs*/;

  params.arraysize = params.nblocks * params.blocksz;

  FMPI_DBG(params);

  return params;
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  std::size_t num_cpus = std::thread::hardware_concurrency() / 2;

  //  constexpr std::size_t l2Cache = 262144;

  constexpr auto max_blocksize = std::size_t(1) << 20;
  // constexpr auto ndispatchers  = 1;

  constexpr std::size_t min_procs = 16;
  constexpr std::size_t max_procs = debug ? min_procs : 64;

  constexpr std::size_t min_blocksz = 256;
  constexpr std::size_t max_blocksz = debug ? min_blocksz : 1 << 20;

  constexpr std::size_t min_winsz = 4;
  constexpr std::size_t max_winsz = 4;
  //constexpr std::size_t max_winsz = debug ? min_winsz : 32;

  for (long np = min_procs; np <= max_procs; np *= 2) {
    for (long block_bytes = min_blocksz; block_bytes <= max_blocksz;
         block_bytes *= 8) {
      long const blocksz = block_bytes / sizeof(value_t);

      for (long ws = min_winsz; ws <= max_winsz; ws *= 2) {
        b->Args({np, blocksz, ws});
      }
    }
  }
}

template <class Iter>
static void random(
    Iter first, Iter last, std::size_t blocksz, std::size_t nblocks) {
#pragma omp parallel default(none) firstprivate(first, last, nblocks, blocksz)
  {
    std::random_device r;
    std::seed_seq      seed_seq{r(), r(), r(), r(), r(), r()};
    std::mt19937_64    generator(seed_seq);
    std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
    for (std::size_t block = 0; block < nblocks; ++block) {
      // generate some random values
      auto bf = std::next(first, block * blocksz);
      auto bl = std::min(std::next(first, (block + 1) * blocksz), last);

      std::generate(bf, bl, [&]() { return distribution(generator); });
      // sort it
      std::sort(bf, bl);
    }
  }
}

using vector_t = std::vector<value_t>;

static void BM_TlxMergeSequential(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  FMPI_DBG("BM_TlxMergeSequential");

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const params = processParams(state);

  vector_t src(params.arraysize);
  vector_t target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), params.blocksz, params.nblocks);

    for (auto&& b : fmpi::range(params.nblocks)) {
      auto f    = std::next(std::begin(src), b * params.blocksz);
      auto l    = std::next(f, params.blocksz);
      chunks[b] = std::make_pair(f, l);
    }

    state.ResumeTiming();

    auto res = tlx::multiway_merge(
        chunks.begin(), chunks.end(), std::begin(target), params.arraysize);

    benchmark::DoNotOptimize(res);

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeSequentialRecursive(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  FMPI_DBG("BM_TlxMergeSequentialRecursive");

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const params = processParams(state);

  vector_t src(params.arraysize);
  vector_t target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.windowsz);
  auto processed =
      std::vector<std::pair<iterator, iterator>>(params.windowsz);

  std::size_t mergedepth = params.nblocks / params.windowsz +
                           ((params.nblocks % params.windowsz) > 0);
  FMPI_DBG(mergedepth);

  auto const window_nels = params.windowsz * params.blocksz;

  FMPI_DBG(window_nels);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), params.blocksz, params.nblocks);

    state.ResumeTiming();

    auto first = std::begin(target);

    for (auto&& level : fmpi::range(mergedepth)) {
      auto w_first = std::next(std::begin(src), level * window_nels);

      auto const max = std::distance(w_first, std::end(src));
      auto       w_last =
          std::next(w_first, std::min<std::size_t>(window_nels, max));

      auto const n            = std::distance(w_first, w_last);
      auto const n_blocks_win = std::max<std::size_t>(n / params.blocksz, 1);

      for (auto&& b : fmpi::range(
               std::min<std::size_t>(params.windowsz, n_blocks_win))) {
        auto f    = std::next(w_first, b * params.blocksz);
        auto l    = std::min(std::next(f, params.blocksz), w_last);
        chunks[b] = std::make_pair(f, l);
      }

      auto last_it =
          tlx::multiway_merge(chunks.begin(), chunks.end(), first, n);

      processed.emplace_back(first, last_it);
      std::swap(first, last_it);
    }

    FMPI_ASSERT(first == std::end(target));
    FMPI_ASSERT(first == std::end(target));

    using simple_vector =
        tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto buffer = simple_vector{params.arraysize};

    auto last_it = tlx::multiway_merge(
        processed.begin(),
        processed.end(),
        std::begin(buffer),
        params.arraysize);

    auto res =
        std::move(std::begin(buffer), std::end(buffer), std::begin(target));

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeParallel(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const nthreads = config.num_threads;

  auto const params = processParams(state);

  FMPI_DBG(nthreads);

  vector_t src(params.arraysize);
  vector_t target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), params.blocksz, params.nblocks);

    for (auto&& b : fmpi::range(params.nblocks)) {
      auto f    = std::next(std::begin(src), b * params.blocksz);
      auto l    = std::next(f, params.blocksz);
      chunks[b] = std::make_pair(f, l);
    }

    state.ResumeTiming();

    auto res = tlx::parallel_multiway_merge(
        chunks.begin(),
        chunks.end(),
        std::begin(target),
        params.arraysize,
        std::less<>{},
        tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
        tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
        nthreads);

    benchmark::DoNotOptimize(res);

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeParallelRecursive(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  auto const  params = processParams(state);
  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const nthreads = config.num_threads;

  FMPI_DBG(nthreads);

  vector_t src(params.arraysize);
  vector_t target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.windowsz);
  auto processed =
      std::vector<std::pair<iterator, iterator>>(params.windowsz);

  std::size_t mergedepth = params.nblocks / params.windowsz +
                           ((params.nblocks % params.windowsz) > 0);
  FMPI_DBG(mergedepth);

  auto const window_nels = params.windowsz * params.blocksz;

  FMPI_DBG(window_nels);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), params.blocksz, params.nblocks);

    state.ResumeTiming();

    auto first = std::begin(target);

    for (auto&& level : fmpi::range(mergedepth)) {
      auto w_first = std::next(std::begin(src), level * window_nels);

      auto const max = std::distance(w_first, std::end(src));
      auto       w_last =
          std::next(w_first, std::min<std::size_t>(window_nels, max));

      auto const n            = std::distance(w_first, w_last);
      auto const n_blocks_win = std::max<std::size_t>(n / params.blocksz, 1);

      for (auto&& b : fmpi::range(
               std::min<std::size_t>(params.windowsz, n_blocks_win))) {
        auto f    = std::next(w_first, b * params.blocksz);
        auto l    = std::min(std::next(f, params.blocksz), w_last);
        chunks[b] = std::make_pair(f, l);
      }

      auto last_it = tlx::parallel_multiway_merge(
          chunks.begin(),
          chunks.end(),
          first,
          n,
          std::less<>{},
          tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
          tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
          nthreads);

      processed.emplace_back(first, last_it);
      std::swap(first, last_it);
    }

    FMPI_ASSERT(first == std::end(target));

    using simple_vector =
        tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto buffer = simple_vector{params.arraysize};

    auto last_it = tlx::parallel_multiway_merge(
        processed.begin(),
        processed.end(),
        std::begin(buffer),
        params.arraysize,
        std::less<>{},
        tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
        tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
        nthreads);

    auto res =
        std::move(std::begin(buffer), std::end(buffer), std::begin(target));

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_StdSort(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const params = processParams(state);

  vector_t src(params.arraysize);
  vector_t target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), params.blocksz, params.nblocks);

    state.ResumeTiming();

    std::sort(std::begin(src), std::end(src));

    auto res = std::copy(std::begin(src), std::end(src), std::begin(target));

    benchmark::DoNotOptimize(res);

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

BENCHMARK(BM_TlxMergeSequential)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_TlxMergeParallel)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_StdSort)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_TlxMergeSequentialRecursive)
    ->Apply(CustomArguments)
    ->UseRealTime();
BENCHMARK(BM_TlxMergeParallelRecursive)
    ->Apply(CustomArguments)
    ->UseRealTime();

// Run the benchmark
// BENCHMARK_MAIN();

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
 public:
  NullReporter() {
  }
  virtual bool ReportContext(const Context&) {
    return true;
  }
  virtual void ReportRuns(const std::vector<Run>&) {
  }
  virtual void Finalize() {
  }
};

// The main is rewritten to allow for MPI initializing and for selecting a
// reporter according to the process rank
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);

  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Single);

  auto const& world = mpi::Context::world();

  if (world.rank() == 0)
    // root process will use a reporter from the usual set provided by
    // ::benchmark
    ::benchmark::RunSpecifiedBenchmarks();
  else {
    // reporting from other processes is disabled by passing a custom
    // reporter
    NullReporter null;
    ::benchmark::RunSpecifiedBenchmarks(&null);
  }

  mpi::finalize();

  return 0;
}
