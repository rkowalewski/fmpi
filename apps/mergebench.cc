#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
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

static void BM_TlxMerge(benchmark::State& state)
{
  using value_t = int;

  auto const nblocks   = state.range(0);
  auto const blocksize = state.range(1) / sizeof(value_t);

  auto const size = nblocks * blocksize;

  std::vector<value_t> src(size);
  std::vector<value_t> target(size);

  std::mt19937_64 generator(random_seed_seq::get_instance());
  std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);

  using iterator = typename std::vector<value_t>::iterator;

  // initialize vectors with random numbers
  std::generate(
      src.begin(), src.end(), [&]() { return distribution(generator); });

  std::vector<std::pair<iterator, iterator>> chunks(nblocks);

  for (auto b = 0; b < nblocks; ++b) {
    auto f    = std::next(src.begin(), b * blocksize);
    auto l    = std::next(f, blocksize);
    std::sort(f, l);
    chunks[b] = std::make_pair(f, l);
  }

  for (auto _ : state) {
#if 0
    auto res = tlx::parallel_multiway_merge(
        chunks.begin(), chunks.end(), target.begin(), size);
#else

    auto res = __gnu_parallel::multiway_merge(
        std::begin(chunks),
        std::end(chunks),
        target.begin(),
        size,
        std::less<>{},
        __gnu_parallel::parallel_tag{0});
#endif

    benchmark::DoNotOptimize(res);
  }
}

static void CustomArguments(benchmark::internal::Benchmark* b)
{
  constexpr long max_blocks = 48;
  for (long i = 6; i <= max_blocks; i *= 2)
    for (long j = 8; j <= 1024 * 1024; j *= 8) b->Args({i, j});
}

BENCHMARK(BM_TlxMerge)->Apply(CustomArguments)->UseRealTime();

// Run the benchmark
BENCHMARK_MAIN();
