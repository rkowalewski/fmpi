#include <mpi.h>

#include <cassert>
#include <fmpi/Debug.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <tlx/algorithm.hpp>

#include "benchmark.hpp"

void CustomArguments(benchmark::internal::Benchmark* b) {
  constexpr long min_procs = 16;
  constexpr long max_procs = debug ? min_procs : 64;

  constexpr long min_blocksz = 256;
  constexpr long max_blocksz = debug ? min_blocksz : 1 << 20;

  for (long np = min_procs; np <= max_procs; np *= 2) {
    for (long block_bytes = min_blocksz; block_bytes <= max_blocksz;
         block_bytes *= 4) {
      constexpr long ws = 0;

      b->Args(
          {np, block_bytes, ws, static_cast<long>(omp_get_max_threads())});
    }
  }
}

static void BM_TlxMergeSequential(benchmark::State& state) {
  using iterator = typename container::iterator;

  FMPI_DBG("BM_TlxMergeSequential");

  auto const params = processParams(state);

  container src(params.arraysize);
  container target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(
        std::begin(src),
        std::end(src),
        std::make_pair(params.blocksz, params.nblocks));

    for (auto&& b : fmpi::range(params.nblocks)) {
      auto f    = std::next(std::begin(src), b * params.blocksz);
      auto l    = std::next(f, params.blocksz);
      chunks[b] = std::make_pair(f, l);
    }

    state.ResumeTiming();

    auto res = tlx::multiway_merge(
        chunks.begin(), chunks.end(), std::begin(target), params.arraysize);

    benchmark::DoNotOptimize(res);

    assert(res == std::end(target));
    assert(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeParallel(benchmark::State& state) {
  using iterator = typename container::iterator;

  auto const& config = fmpi::Pinning::instance();

  auto const nthreads = config.num_threads;

  auto const params = processParams(state);

  FMPI_DBG(nthreads);

  container src(params.arraysize);
  container target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(
        std::begin(src),
        std::end(src),
        std::make_pair(params.blocksz, params.nblocks));

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

    assert(res == std::end(target));
    assert(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_StdSort(benchmark::State& state) {
  using iterator = typename container::iterator;

  auto const params = processParams(state);

  container src(params.arraysize);
  container target(params.arraysize);

  auto chunks = std::vector<std::pair<iterator, iterator>>(params.nblocks);

  for (auto _ : state) {
    state.PauseTiming();

    random(
        std::begin(src),
        std::end(src),
        std::make_pair(params.blocksz, params.nblocks));

    state.ResumeTiming();

    std::sort(std::begin(src), std::end(src));

    auto res = std::copy(std::begin(src), std::end(src), std::begin(target));

    benchmark::DoNotOptimize(res);

    assert(res == std::end(target));
    assert(std::is_sorted(std::begin(target), std::end(target)));
  }
}

BENCHMARK(BM_TlxMergeSequential)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_TlxMergeParallel)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_StdSort)->Apply(CustomArguments)->UseRealTime();
