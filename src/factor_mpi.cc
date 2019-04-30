#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include "Timer.h"
#include "random.h"

#include <synchronized_barrier.hpp>

using rank_pair = std::pair<int, int>;

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

template <class T>
inline constexpr T mod(T a, T b)
{
  assert(b > 0);
  T ret = a % b;
  return (ret >= 0) ? (ret) : (ret + b);
}

inline constexpr rank_pair oneFactor_classic(int me, int r, int n)
{
  auto k_bottom = 2 * n - 1;
  return std::make_pair(mod(r + me, k_bottom) + 1, mod(r - me, k_bottom) + 1);
}

template <class InputIt, class OutputIt>
inline void factorParty(InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  assert(nr % 2 == 0);

  // We have 2n ranks
  auto n = nr / 2;

  std::vector<int> partner;
  partner.reserve(nr);

  // Rounds
  for (int r = 1; r < nr; ++r) {
    partner[0] = r;
    partner[r] = 0;

    // generate remaining pairs
    for (int p = 1; p < n; ++p) {
      auto pair            = oneFactor_classic(p, r - 1, n);
      partner[pair.first]  = pair.second;
      partner[pair.second] = pair.first;
    }

    auto res = MPI_Sendrecv(
        std::addressof(*(begin + partner[me] * nels)),
        nels,
        MPI_INT,
        partner[me],
        100,
        std::addressof(*(out + partner[me] * nels)),
        nels,
        MPI_INT,
        partner[me],
        100,
        comm,
        MPI_STATUS_IGNORE);
    assert(res == MPI_SUCCESS);
  }

  std::copy(begin + me * nels, begin + me * nels + nels, out + me * nels);
}

template <class InputIt, class OutputIt>
inline void flatFactor(InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  for (int i = 1; i <= nr; ++i) {
    auto partner = mod(i - me, nr);
    auto res     = MPI_Sendrecv(
        std::addressof(*(begin + partner * nels)),
        nels,
        MPI_INT,
        partner,
        100,
        std::addressof(*(out + partner * nels)),
        nels,
        MPI_INT,
        partner,
        100,
        comm,
        MPI_STATUS_IGNORE);
    assert(res == MPI_SUCCESS);
  }
}

template <class InputIt, class OutputIt>
inline void flatHandshake(
    InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);
  for (int i = 1; i < nr; ++i) {
    auto pair = std::make_pair(mod(me + i, nr), mod(me - i, nr));
    auto res  = MPI_Sendrecv(
        std::addressof(*(begin + pair.first * nels)),
        nels,
        MPI_INT,
        pair.first,
        100,
        std::addressof(*(out + pair.second * nels)),
        nels,
        MPI_INT,
        pair.second,
        100,
        comm,
        MPI_STATUS_IGNORE);
    assert(res == MPI_SUCCESS);
  }

  std::copy(begin + me * nels, begin + me * nels + nels, out + me * nels);
}

template <class InputIt, class OutputIt>
inline void MpiAlltoAll(InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  auto res = MPI_Alltoall(
      std::addressof(*begin),
      nels,
      MPI_INT,
      std::addressof(*out),
      nels,
      MPI_INT,
      MPI_COMM_WORLD);

  assert(res == MPI_SUCCESS);
}

template <class InputIt, class Gen>
inline void random_data(InputIt begin, InputIt end, Gen gen)
{
  std::generate(begin, end, gen);
}

template <class InputIt>
void printVector(InputIt begin, InputIt end, int me)
{
  using value_t = typename std::iterator_traits<InputIt>::value_type;

  std::ostringstream os;
  os << "rank " << me << ": ";
  std::copy(begin, end, std::ostream_iterator<value_t>(os, " "));
  os << "\n";
  std::cout << os.str();
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
    F&& f, InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  auto start = ChronoClockNow();
  f(begin, out, nels, comm);
  return ChronoClockNow() - start;
}

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters   = 3;
constexpr size_t minbytes = 256 * KB;
constexpr size_t maxbytes = 16 * MB;

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

int main(int argc, char* argv[])
{
  constexpr int n_ranks = 6;

  using value_t     = int;
  using container_t = std::vector<value_t>;
  using iterator_t  = typename container_t::iterator;

  container_t data, out, correct;

  MPI_Init(&argc, &argv);
  int      me, nr;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  if (me == 0) {
    print_env();
  }

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  assert(is_clock_synced);

  std::mt19937_64 generator(rko::random_seed_seq::get_instance());

  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  measurements_t measurements;

  using function_t = std::function<void(
      typename container_t::iterator,
      typename container_t::iterator,
      int,
      MPI_Comm)>;

  function_t f = MpiAlltoAll<
      typename container_t::iterator,
      typename container_t::iterator>;

  std::array<std::pair<std::string, function_t>, 4> algos2 = {
      std::make_pair("AlltoAll", MpiAlltoAll<iterator_t, iterator_t>),
      std::make_pair("FactorParty", factorParty<iterator_t, iterator_t>),
      std::make_pair("FlatFactor", flatFactor<iterator_t, iterator_t>),
      std::make_pair("FlatHandshake", flatHandshake<iterator_t, iterator_t>)};

  auto n_sizes = std::log2(maxbytes / minbytes);

  for (size_t stage = 0; stage <= n_sizes; ++stage) {
    auto n_l_elems = minbytes * (1 << stage) / sizeof(value_t);
    auto n_g_elems = size_t(nr) * n_l_elems;

    data.resize(n_g_elems);
    out.resize(n_g_elems);
    correct.resize(n_g_elems);

    int  idx = 0;
    auto gen = [me, nr, &idx]() { return me * nr + idx++; };

    std::generate(std::begin(data), std::end(data), gen);

    for (size_t it = 0; it < niters; ++it) {
      std::shuffle(std::begin(data), std::end(data), generator);

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      MPI_Alltoall(
          std::addressof(*std::begin(data)),
          n_l_elems,
          MPI_INT,
          std::addressof(*std::begin(correct)),
          n_l_elems,
          MPI_INT,
          MPI_COMM_WORLD);

      for (auto& algo : algos2) {
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(comm);
        assert(barrier.Success(comm));

        auto t = run_algorithm(
            algo.second,
            std::begin(data),
            std::begin(out),
            n_l_elems,
            MPI_COMM_WORLD);

        measurements[algo.first].emplace_back(t);

        assert(std::equal(
            std::begin(correct), std::end(correct), std::begin(out)));
      }
    }

    std::vector<StringDoublePair> ranking;

    constexpr int root = 0;

    for (auto const& algo : algos2) {
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
      assert(ranking.size() == algos2.size());
      // sort the median vector
      std::sort(ranking.begin(), ranking.end());

      std::cout << "(" << stage << ") Global Volume (KB) "
                << n_g_elems * sizeof(value_t) / KB
                << ", Message Size per Proc (KB) = "
                << n_l_elems * sizeof(value_t) / KB << ", ranking: ";
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
