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

#include <synchronized_barrier.hpp>

template <class InputIt>
void printVector(InputIt begin, InputIt end, int me);

struct StringDoublePair : std::pair<std::string, double> {
  using std::pair<std::string, double>::pair;
};

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs)
{
  return lhs.second < rhs.second;
}

std::ostream& operator<<(std::ostream& os, StringDoublePair const& p)
{
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}

template <class InputIt, class Gen>
inline void random_data(InputIt begin, InputIt end, Gen gen)
{
  std::generate(begin, end, gen);
}

auto medianReduce(double myMedian, int root, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);
  std::vector<double> meds;

  meds.reserve(nr);

  MPI_Gather(&myMedian, 1, MPI_DOUBLE, &meds[0], 1, MPI_DOUBLE, root, comm);

  if (me == root) {
    auto nth = &meds[0] + nr / 2;
    std::nth_element(&meds[0], nth, &meds[0] + nr);
    return *nth;
  }
  else {
    return -1.0;
  }
}

template <class InputIt, class OutputIt, class F>
auto run_algorithm(
    F&& f, InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  auto start = ChronoClockNow();
  f(begin, out, blocksize, comm);
  return ChronoClockNow() - start;
}

extern char** environ;

void print_env()
{
  int   i          = 1;
  char* env_var_kv = *environ;

  for (; env_var_kv != 0; ++i) {
    // Split into key and value:
    char*       flag_name_cstr  = env_var_kv;
    char*       flag_value_cstr = std::strstr(env_var_kv, "=");
    int         flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if (std::strstr(flag_name.c_str(), "OMPI_") ||
        std::strstr(flag_name.c_str(), "I_MPI_")) {
      std::cout << flag_name << " = " << flag_value << "\n";
    }

    env_var_kv = *(environ + i);
  }
}

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters       = 10;
constexpr size_t minblocksize = 1 * KB;

// This are approximately 25 GB
// constexpr size_t capacity_per_node = (size_t(1) << 25) * 28 * 28;
constexpr size_t capacity_per_node = 32 * MB;

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

  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  int         me, nr;
  container_t data, out, correct;

  measurements_t measurements;

#if 0
  std::array<std::pair<std::string, benchmark_t>, 5> algos = {
      std::make_pair("AlltoAll", MpiAlltoAll<iterator_t, iterator_t>),
      std::make_pair("FactorParty", factorParty<iterator_t, iterator_t>),
      std::make_pair("FlatFactor", flatFactor<iterator_t, iterator_t>),
      std::make_pair("FlatHandshake", flatHandshake<iterator_t, iterator_t>),
      std::make_pair("Bruck", alltoall_bruck<iterator_t, iterator_t>),
      std::make_pair("Bruck_Mod", alltoall_bruck_mod<iterator_t, iterator_t>)
      };
#else
  std::array<std::pair<std::string, benchmark_t>, 1> algos = {std::make_pair(
      "Bruck_Mod", alltoall_bruck_mod<iterator_t, iterator_t>)};
#endif

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  ASSERT(nr >= 1);

  if (me == 0) {
    print_env();
  }

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  ASSERT(is_clock_synced);

  std::mt19937_64 generator(rko::random_seed_seq::get_instance());

#ifdef NDEBUG
  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  const size_t number_nodes   = 1;
  const size_t procs_per_node = nr / number_nodes;

  const size_t maxblocksize =
      capacity_per_node / (2 * procs_per_node * procs_per_node);
  auto n_sizes = std::log2(maxblocksize / minblocksize);
#else
  size_t                                             n_sizes = 1;
#endif

  for (size_t step = 0; step <= n_sizes; ++step) {
#ifdef NDEBUG
    auto blocksize =
        minblocksize * (1 << step) / (sizeof(value_t) * number_nodes);
#else
    auto blocksize = 1;
#endif

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

        printVector(out.begin(), out.end(), me);

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

  if (me == 0) {
    for (int i = 0; i < 2; ++i) {
      std::cout << i << "\n";
    }
  }
  MPI_Finalize();

  return 0;
}
