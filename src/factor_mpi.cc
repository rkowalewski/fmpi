#include <algorithm>
#include <cassert>
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

std::ostream& operator<<(std::ostream& os, std::pair<int, int> const& p)
{
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}

std::ostream& operator<<(
    std::ostream& os, std::pair<std::string, double> const& p)
{
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}

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

    MPI_Sendrecv(
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
  }

  std::copy(begin + me * nels, begin + me * nels + nels, out + me * nels);
}

template <class InputIt, class OutputIt>
inline void flatFactor_selfLoops(
    InputIt begin, OutputIt out, int nels, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);

  for (int i = 1; i <= nr; ++i) {
    auto partner = mod(i - me, nr);
    MPI_Sendrecv(
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
    MPI_Sendrecv(
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
  }

  std::copy(begin + me * nels, begin + me * nels + nels, out + me * nels);
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
#if 0
    std::cout << "medians: ";
    std::copy(
        &meds[0], &meds[nr], std::ostream_iterator<double>(std::cout, " "));
    std::cout << "\n";
#endif
    auto nth = &meds[0] + nr / 2;
    std::nth_element(&meds[0], nth, &meds[0] + nr);
    return *nth;
  }
  else {
    return -1.0;
  }
}

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters   = 3;
constexpr size_t minbytes = 256 * KB;
constexpr size_t maxbytes = 512 * KB;

int main(int argc, char* argv[])
{
  constexpr int n_ranks = 6;
  using value_t         = int;

  std::vector<int> data, out, correct;

  MPI_Init(&argc, &argv);
  int      me, nr;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  assert(is_clock_synced);

  std::mt19937_64 generator(rko::random_seed_seq::get_instance());

  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  measurements_t measurements;

  std::array<std::string, 4> algos = {
      "AlltoAll", "FactorParty", "FlatFactor", "FlatHandshake"};

  std::vector<std::vector<StringDoublePair>> results;
  // results.reserve(niters);

  auto n_sizes = std::log2(maxbytes / minbytes);

  if (me == 0) std::cout << "sizes: " << n_sizes << "\n";

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

      auto barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));

      auto now = ChronoClockNow();
      MPI_Alltoall(
          std::addressof(*std::begin(data)),
          n_l_elems,
          MPI_INT,
          std::addressof(*std::begin(correct)),
          n_l_elems,
          MPI_INT,
          MPI_COMM_WORLD);
      auto duration = ChronoClockNow() - now;
      measurements["AlltoAll"].emplace_back(duration);

      barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));

      now = ChronoClockNow();
      factorParty(
          std::begin(data), std::begin(out), n_l_elems, MPI_COMM_WORLD);
      duration = ChronoClockNow() - now;
      measurements["FactorParty"].emplace_back(duration);

      assert(std::equal(
          std::begin(correct), std::end(correct), std::begin(out)));

      barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));

      now = ChronoClockNow();
      flatFactor_selfLoops(
          std::begin(data), std::begin(out), n_l_elems, MPI_COMM_WORLD);
      duration = ChronoClockNow() - now;
      measurements["FlatFactor"].emplace_back(duration);

      assert(std::equal(
          std::begin(correct), std::end(correct), std::begin(out)));

      barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));

      now = ChronoClockNow();
      flatHandshake(
          std::begin(data), std::begin(out), n_l_elems, MPI_COMM_WORLD);
      duration = ChronoClockNow() - now;
      measurements["FlatHandshake"].emplace_back(duration);

      assert(std::equal(
          std::begin(correct), std::end(correct), std::begin(out)));

      barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));

      barrier = clock.Barrier(comm);
      assert(barrier.Success(comm));
    }

    std::vector<StringDoublePair> ranking;

    for (auto const& algo : algos) {
      auto mid = (niters / 2);

      std::nth_element(
          measurements[algo].begin(),
          measurements[algo].begin() + mid,
          measurements[algo].end());

      auto med = medianReduce(measurements[algo][mid], 0, comm);

      if (me == 0) {
        ranking.emplace_back(StringDoublePair{algo, med});
      }
    }

    if (me == 0) {
      std::sort(ranking.begin(), ranking.end());
      assert(ranking.size() == algos.size());
      results.emplace_back(std::move(ranking));
      assert(results.size() == (stage + 1));

#if 0
      auto& res = results[stage];

      std::cout << "(" << stage
                << ") KB = " << n_g_elems * sizeof(value_t) / KB
                << " medians: ";
      std::copy(
          res.begin(),
          res.end(),
          std::ostream_iterator<StringDoublePair>(std::cout, " "));
      std::cout << "\n";
#endif
    }

#if 0
    if (me == 0) {
      std::cout << "KB: " << n_g_elems * sizeof(value_t) / KB << "\n";
      for (auto& kv : measurements) {
        std::cout << kv.first << ": ";
        std::copy(
            std::begin(kv.second),
            std::end(kv.second),
            std::ostream_iterator<double>(std::cout, ", "));
        std::cout << "\n";

        std::cout << "2nd element: " << kv.second[1] << "\n";
      }

      std::cout << "\n";
    }
#endif

    measurements.clear();
  }

  if (me == 0) {
    for (std::size_t i = 0; i <= n_sizes; ++i) {
      std::cout << "(" << i << ") KB = " << minbytes * (1 << i) * nr / KB
                << " medians: ";
      std::copy(
          std::begin(results[i]),
          std::end(results[i]),
          std::ostream_iterator<StringDoublePair>(std::cout, ", "));
      std::cout << "\n";
    }
  }

  MPI_Finalize();

  return 0;
}
