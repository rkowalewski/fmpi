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

static void CustomArguments(benchmark::internal::Benchmark* b) {
  std::size_t num_cpus = std::thread::hardware_concurrency() / 2;

  //  constexpr std::size_t l2Cache = 262144;

  constexpr auto max_blocksize = std::size_t(1) << 20;
  // constexpr auto ndispatchers  = 1;

  for (std::size_t block_bytes = 256; block_bytes <= max_blocksize;
       block_bytes *= 8) {
    long const blocksz = block_bytes / sizeof(value_t);

    for (long ws = 16; ws <= 16; ws *= 2) {
      b->Args({blocksz, ws});
    }
  }
}

template <class Iter>
static void random(
    Iter first, Iter last, std::size_t blocksz, std::size_t nblocks) {
  auto const nels = std::size_t(std::distance(first, last));

  FMPI_ASSERT(nels / blocksz == nblocks);
  FMPI_ASSERT(nels % blocksz == 0);

#pragma omp parallel default(none) firstprivate(first, last, nblocks, blocksz)
  {
    std::random_device r;
    std::seed_seq      seed_seq{r(), r(), r(), r(), r(), r()};
    std::mt19937_64    generator(seed_seq);
    std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
    for (std::size_t block = 0; block < std::size_t(nblocks); ++block) {
      // generate some random values
      auto bf = std::next(first, block * blocksz);
      auto bl = std::next(first, (block + 1) * blocksz);

      std::generate(bf, bl, [&]() { return distribution(generator); });
      // sort it
      std::sort(bf, bl);
    }
  }
}

using vector_t = std::vector<value_t>;

static void BM_TlxMergeSequential(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  auto const blocksize = state.range(0);
  auto const windowsz  = state.range(1);

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const nblocks = world.size();

  auto const size = nblocks * blocksize;

  FMPI_DBG(nblocks);
  FMPI_DBG(size);

  vector_t src(size);
  vector_t target(size);

  auto chunks = std::vector<std::pair<iterator, iterator>>(nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), blocksize, nblocks);

    for (auto&& b : fmpi::range(nblocks)) {
      auto f    = std::next(std::begin(src), b * blocksize);
      auto l    = std::next(f, blocksize);
      chunks[b] = std::make_pair(f, l);
    }

    state.ResumeTiming();

    auto res = tlx::multiway_merge(
        chunks.begin(), chunks.end(), std::begin(target), size);

    benchmark::DoNotOptimize(res);

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeParallel(benchmark::State& state) {
  using iterator = typename vector_t::iterator;

  auto const blocksize = state.range(0);
  auto const windowsz  = state.range(1);

  auto const& world  = mpi::Context::world();
  auto const& config = fmpi::Config::instance();

  auto const nblocks  = world.size();
  auto const nthreads = config.num_threads;

  auto const size = nblocks * blocksize;

  FMPI_DBG(nblocks);
  FMPI_DBG(size);
  FMPI_DBG(nthreads);

  vector_t src(size);
  vector_t target(size);

  auto chunks = std::vector<std::pair<iterator, iterator>>(nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(std::begin(src), std::end(src), blocksize, nblocks);

    for (auto&& b : fmpi::range(nblocks)) {
      auto f    = std::next(std::begin(src), b * blocksize);
      auto l    = std::next(f, blocksize);
      chunks[b] = std::make_pair(f, l);
    }

    state.ResumeTiming();

    auto res = tlx::parallel_multiway_merge(
        chunks.begin(),
        chunks.end(),
        std::begin(target),
        size,
        std::less<>{},
        tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
        tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
        nthreads);

    benchmark::DoNotOptimize(res);

    FMPI_ASSERT(res == std::end(target));
    FMPI_ASSERT(std::is_sorted(std::begin(target), std::end(target)));
  }
}

#if 0
static void CustomArguments(benchmark::internal::Benchmark* b) {
  // std::size_t num_threads = omp_get_max_threads();

  // std::size_t num_nodes = 1,2,4,8,16,32,64
  // std::size_t num_procs = 1,2,4,8
  // std::size_t num_threads = std::thread::hardware_concurrency() / 2 /
  // num_procs std::size_t window_sizes = 1,2,4,8,16,32
  std::size_t num_cpus = std::thread::hardware_concurrency() / 2;

  // constexpr std::size_t l1dCache = 32768;
  constexpr std::size_t l2Cache = 262144;
  // constexpr std::size_t l3Cache  = 2883584;

#if 0
  std::vector<int> cpu_names(num_threads);

#pragma omp parallel default(none) shared(cpu_names)
  {
    auto thread_num       = omp_get_thread_num();  // Test
    int  cpu              = sched_getcpu();
    cpu_names[thread_num] = cpu;
  }

  for (auto&& r : fmpi::range(cpu_names.size())) {
    std::cout << "{" << r << ", " << cpu_names[r] << "}\n";
  }
#endif

  constexpr auto max_blocksize = std::size_t(1) << 20;
  constexpr auto ndispatchers  = 1;

  for (std::size_t nn = 64; nn <= 64; nn *= 2) {
    for (std::size_t np = 4; np <= 4; np *= 2) {
      for (std::size_t ws = 16; ws <= 16; ws *= 2) {
        for (std::size_t bytes = sizeof(value_t); bytes <= max_blocksize;
             bytes *= 8) {
          long const blocksz = bytes / sizeof(value_t);
          long const nblocks = nn * np;
          long const nthreads =
              (std::thread::hardware_concurrency() / (2 * np)) - ndispatchers;

          omp_set_num_threads(static_cast<int>(nthreads));

          b->Args({nblocks, blocksz, nthreads});
        }
      }
    }
  }

#if 0
  for (long i = 1; i <= long(num_cpus); i *= 2) {
    long const max = l2Cache * i;
    for (long j = 8; j <= max; j *= 8) {
      b->Args({omp_get_max_threads(), j});
    }
  }
#endif
}

static void BM_Sort(benchmark::State& state) {
  auto const nblocks   = state.range(0);
  auto const size      = state.range(1) / sizeof(value_t);
  auto const blocksize = size / nblocks;

  std::vector<value_t> src(size);
  std::vector<value_t> target(size);

  random(std::begin(src), std::end(src));

  using iterator = typename std::vector<value_t>::iterator;

  std::vector<std::pair<iterator, iterator>> chunks(nblocks);

  for (auto _ : state) {
    for (auto b = 0; b < nblocks; ++b) {
      auto f = std::next(src.begin(), b * blocksize);
      auto l = std::next(f, blocksize);
      std::sort(f, l);
      chunks[b] = std::make_pair(f, l);
    }
    std::sort(src.begin(), src.end());
    auto res = std::copy(src.begin(), src.end(), target.begin());
    benchmark::DoNotOptimize(res);
  }
}
#endif

BENCHMARK(BM_TlxMergeSequential)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_TlxMergeParallel)->Apply(CustomArguments)->UseRealTime();
// BENCHMARK(BM_Sort)->Apply(CustomArguments)->UseRealTime();

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
