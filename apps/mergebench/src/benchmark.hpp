#ifndef MERGEBENCH_BENCHMARK_HPP
#define MERGEBENCH_BENCHMARK_HPP

#include <benchmark/benchmark.h>

#include <cstddef>
#include <fmpi/Debug.hpp>
#include <iosfwd>
#include <random>
#include <vector>

using value_t   = double;
using container = std::vector<value_t>;

int main(int argc, char** argv);

void CustomArguments(benchmark::internal::Benchmark* b);

static constexpr bool debug =
#ifndef NDEBUG
    true
#else
    false
#endif
    ;

struct Params {
  std::size_t nprocs{};
  std::size_t nblocks{};
  std::size_t blocksz{};
  std::size_t windowsz{};
  std::size_t arraysize{};
};

Params processParams(benchmark::State const& state);

std::ostream& operator<<(std::ostream& os, Params const& p);

template <class T>
class RandomBetween {
  using uniform_dist = std::conditional_t<
      std::is_integral_v<T>,
      std::uniform_int_distribution<T>,
      std::uniform_real_distribution<T>>;

 public:
  RandomBetween()
    : RandomBetween(
          std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {
  }

  RandomBetween(T low, T high)
    : random_engine_{std::random_device{}()}
    , distribution_{low, high} {
  }

  T operator()() {
    return distribution_(random_engine_);
  }

 private:
  std::mt19937 random_engine_;
  uniform_dist distribution_;
};

using block_spec = std::pair<std::size_t, std::size_t>;

template <
    class Iter,
    class Generator =
        RandomBetween<typename std::iterator_traits<Iter>::value_type>>
static void random(
    Iter first, Iter last, block_spec spec, Generator&& gen = Generator()) {
  // #pragma omp parallel default(none) firstprivate(first, last, spec, gen)
  {
    auto const nblocks = spec.second;
    auto const blocksz = spec.first;
    //#pragma omp for
    for (std::size_t block = 0; block < nblocks; ++block) {
      // generate some random values
      auto bf = std::min(std::next(first, block * blocksz), last);
      auto bl = std::min(std::next(first, (block + 1) * blocksz), last);

      std::generate(bf, bl, gen);

      // sort it
      std::sort(bf, bl);
    }
  }
}

#endif
