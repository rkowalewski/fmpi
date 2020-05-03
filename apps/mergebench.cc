#include <benchmark/benchmark.h>

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <fmpi/Config.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Random.hpp>
#include <iostream>
#include <parallel/algorithm>
#include <random>
#include <sstream>
#include <thread>
#include <tlx/algorithm.hpp>
#include <vector>

#if 0

void callee(int i)
{
  (void)i;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

static void MyMain(int size)
{
#pragma omp parallel for
  for (int i = 0; i < size; i++) callee(i);
}

static void BM_OpenMP(benchmark::State& state)
{
  for (auto _ : state) MyMain(state.range(0));
}

// Measure the time spent by the main thread, use it to decide for how long to
// run the benchmark loop. Depending on the internal implementation detail may
// measure to anywhere from near-zero (the overhead spent before/after work
// handoff to worker thread[s]) to the whole single-thread time.
BENCHMARK(BM_OpenMP)->Range(8, 8 << 10);

// Measure the user-visible time, the wall clock (literally, the time that
// has passed on the clock on the wall), use it to decide for how long to
// run the benchmark loop. This will always be meaningful, an will match the
// time spent by the main thread in single-threaded case, in general
// decreasing with the number of internal threads doing the work.
BENCHMARK(BM_OpenMP)->Range(8, 8 << 10)->UseRealTime();

// Measure the total CPU consumption, use it to decide for how long to
// run the benchmark loop. This will always measure to no less than the
// time spent by the main thread in single-threaded case.
BENCHMARK(BM_OpenMP)->Range(8, 8 << 10)->MeasureProcessCPUTime();

// A mixture of the last two. Measure the total CPU consumption, but use the
// wall clock to decide for how long to run the benchmark loop.
BENCHMARK(BM_OpenMP)
    ->Range(8, 8 << 10)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
#endif

template <class Iter>
void random(Iter first, Iter last) {
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r()};
  std::mt19937_64    generator(seed_seq);

  using value_t = typename std::iterator_traits<Iter>::value_type;

  std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);

  // initialize vectors with random numbers
  std::generate(
      first, last, [&]() { return distribution(generator); });
}

static void BM_TlxMergeSequential(benchmark::State& state) {
  using value_t = int;

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
    auto res = tlx::parallel_multiway_merge(
        chunks.begin(), chunks.end(), target.begin(), size);
    benchmark::DoNotOptimize(res);
  }
}

static void BM_TlxMerge(benchmark::State& state) {
  using value_t = int;

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
    auto res = tlx::parallel_multiway_merge(
        chunks.begin(), chunks.end(), target.begin(), size);
    benchmark::DoNotOptimize(res);
  }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  // std::size_t num_threads = omp_get_max_threads();
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

  for (long i = 1; i <= long(num_cpus); i *= 2) {
    long const max = l2Cache * i;
    for (long j = 8; j <= max; j *= 8) {
      b->Args({omp_get_max_threads(), j});
    }
  }
}

static void BM_Sort(benchmark::State& state) {
  using value_t = int;

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

BENCHMARK(BM_TlxMerge)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_TlxMergeSequential)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_Sort)->Apply(CustomArguments)->UseRealTime();

// Run the benchmark
BENCHMARK_MAIN();
