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

#include <Bruck.h>
#include <Debug.h>
#include <Random.h>
#include <Timer.h>
#include <Types.h>
#include <Benchmark.h>

#include <synchronized_barrier.hpp>

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters       = 10;
constexpr size_t minblocksize = 1 * KB;

// This are approximately 25 GB
constexpr size_t capacity_per_node = (size_t(1) << 25) * 28 * 28;

int main(int argc, char* argv[])
{
  using value_t     = int;
  using container_t = std::vector<value_t>;
  using iterator_t  = typename container_t::iterator;

  using benchmark_t = std::function<void(
      typename container_t::iterator,
      typename container_t::iterator,
      int,
      MPI_Comm)>;
#if 1
  std::array<std::pair<std::string, benchmark_t>, 6> algos = {
      std::make_pair("AlltoAll", MpiAlltoAll<iterator_t, iterator_t>),
      std::make_pair("FactorParty", factorParty<iterator_t, iterator_t>),
      std::make_pair("FlatFactor", flatFactor<iterator_t, iterator_t>),
      std::make_pair("FlatHandshake", flatHandshake<iterator_t, iterator_t>),
      std::make_pair("Bruck", alltoall_bruck<iterator_t, iterator_t>),
      std::make_pair("Bruck_Mod", alltoall_bruck_mod<iterator_t, iterator_t>)
      };
#else
  std::array<std::pair<std::string, benchmark_t>, 2> algos = {std::make_pair(
      "Bruck", alltoall_bruck<iterator_t, iterator_t>),
      std::make_pair("Bruck_Mod", alltoall_bruck_mod<iterator_t, iterator_t>)};
#endif

  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  int         me, nr;
  container_t data, out, correct;

  measurements_t measurements;


  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  if (argc == 1) {
    if (me == 0) {
      std::cout << "usage: " << argv[0] << " [number of nodes]\n";
    }
    MPI_Finalize();
    return 1;
  }


  ASSERT(nr >= 1);

  auto nnodes = std::atoi(argv[1]);

  if (me == 0) {
    print_env();
  }

  std::mt19937_64 generator(random_seed_seq::get_instance());

  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  ASSERT((nr % nnodes) == 0);

  auto procs_per_node = nr / nnodes;

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  ASSERT(is_clock_synced);

  const size_t maxblocksize =
      capacity_per_node / (2 * procs_per_node * procs_per_node);
  auto n_sizes = std::log2(maxblocksize / minblocksize);

  for (size_t step = 0; step <= n_sizes; ++step) {
    auto blocksize =
        minblocksize * (1 << step) / (sizeof(value_t) * nnodes);

    // Required by good old 32-bit MPI
    ASSERT(blocksize > 0 && blocksize < std::numeric_limits<int>::max());

    auto n_g_elems = size_t(nr) * blocksize;

    data.resize(n_g_elems);
    out.resize(n_g_elems);
#ifndef NDEBUG
    correct.resize(n_g_elems);
#endif

    std::iota(std::begin(data), std::end(data), me * nr);

    for (size_t it = 0; it < niters; ++it) {
#ifdef NDEBUG
      std::shuffle(std::begin(data), std::end(data), generator);
#endif

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      constexpr auto mpi_datatype = mpi::mpi_datatype<
          typename std::iterator_traits<iterator_t>::value_type>::value;
      auto res = MPI_Alltoall(
          std::addressof(*std::begin(data)),
          blocksize,
          mpi_datatype,
          std::addressof(*std::begin(correct)),
          blocksize,
          mpi_datatype,
          MPI_COMM_WORLD);
#else
      auto res = MPI_SUCCESS;
#endif

      ASSERT(res == MPI_SUCCESS);

      for (auto const& algo : algos) {
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(comm);
        ASSERT(barrier.Success(comm));

        auto t = run_algorithm(
            algo.second,
            std::begin(data),
            std::begin(out),
            blocksize,
            MPI_COMM_WORLD);

        measurements[algo.first].emplace_back(t);

        ASSERT(std::equal(
            std::begin(correct), std::end(correct), std::begin(out)));
      }
    }

    std::vector<StringDoublePair> ranking;

    constexpr int root = 0;

    for (auto const& algo : algos) {
      auto mid = (niters / 2);

      auto& results = measurements[algo.first];

      // local median
      std::nth_element(results.begin(), results.begin() + mid, results.end());

      // global median
      auto med = medianReduce(results[mid], root, comm);

      // collect the global median into a vector
      if (me == root) {
        ranking.emplace_back(StringDoublePair{algo.first, med});
      }
    }

    if (me == root) {
      ASSERT(ranking.size() == algos.size());
      // sort the median vector
      std::sort(ranking.begin(), ranking.end());

      std::cout << "(" << step << ") Global Volume (KB) "
                << n_g_elems * nr * sizeof(value_t) / KB
                << ", Blocksize (KB) = " << blocksize * sizeof(value_t) / KB
                << ", ranking: ";
      // print until second to last
      std::copy(
          std::begin(ranking),
          std::end(ranking) - 1,
          std::ostream_iterator<StringDoublePair>(std::cout, ", "));
      // print last
      std::cout << *(std::prev(ranking.end()));

      // flush stdio buffer
      std::cout << std::endl;
    }

    // reset measurements for next iteration
    measurements.clear();
  }

  MPI_Finalize();

  return 0;
}



