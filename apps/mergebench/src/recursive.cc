#include "benchmark.hpp"

#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <tlx/algorithm.hpp>
#include <tlx/simple_vector.hpp>

void CustomArguments(benchmark::internal::Benchmark* b) {
  constexpr long min_procs = 16;
  constexpr long max_procs = debug ? min_procs : 64;

  constexpr long min_blocksz = 256;
  constexpr long max_blocksz = debug ? min_blocksz : 1 << 20;

  constexpr long min_winsz = 4;
  constexpr long max_winsz = debug ? min_winsz : 32;

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

static void BM_TlxMergeSequentialRecursive(benchmark::State& state) {
  using iterator = typename container::iterator;

  auto const params = processParams(state);

  container src(params.arraysize);
  container target(params.arraysize);

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

    random(
        std::begin(src),
        std::end(src),
        std::make_pair(params.blocksz, params.nblocks));

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

    assert(first == std::end(target));
    assert(first == std::end(target));

    using simple_vector =
        tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto buffer = simple_vector{params.arraysize};

    auto last_it = tlx::multiway_merge(
        processed.begin(),
        processed.end(),
        std::begin(buffer),
        params.arraysize);

    assert(last_it == std::end(buffer));

    auto res =
        std::move(std::begin(buffer), std::end(buffer), std::begin(target));

    benchmark::DoNotOptimize(res);

    assert(res == std::end(target));
    assert(std::is_sorted(std::begin(target), std::end(target)));
  }
}

static void BM_TlxMergeParallelRecursive(benchmark::State& state) {
  using iterator = typename container::iterator;

  auto const  params = processParams(state);
  auto const& config = fmpi::Pinning::instance();

  auto const nthreads = config.num_threads;

  FMPI_DBG(nthreads);

  container src(params.arraysize);
  container target(params.arraysize);

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

    random(
        std::begin(src),
        std::end(src),
        std::make_pair(params.blocksz, params.nblocks));

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

    assert(first == std::end(target));

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

    assert(last_it == std::end(buffer));

    auto res =
        std::move(std::begin(buffer), std::end(buffer), std::begin(target));

    assert(res == std::end(target));
    assert(std::is_sorted(std::begin(target), std::end(target)));

    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_TlxMergeSequentialRecursive)
    ->Apply(CustomArguments)
    ->UseRealTime();
BENCHMARK(BM_TlxMergeParallelRecursive)
    ->Apply(CustomArguments)
    ->UseRealTime();
