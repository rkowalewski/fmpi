#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <sstream>
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

struct PAIR : std::pair<std::string, double> {
  using std::pair<std::string, double>::pair;
};

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

int main(int argc, char* argv[])
{
  constexpr int n_ranks = 6;
  using value_t         = int;

  constexpr size_t KB = (1 << 10) / sizeof(value_t);
  constexpr size_t MB = (KB << 10);

  // constexpr size_t sz_begin = 16 * KB;
  constexpr size_t sz_begin = 256 * KB;

  constexpr size_t sz_max = 128 * MB;

  std::vector<int>  data, out, correct;
  std::vector<PAIR> times;

  MPI_Init(&argc, &argv);
  int      me, nr;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  int  idx = 0;
  auto gen = [me, nr, &idx]() { return me * nr + idx++; };

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  assert(is_clock_synced);

  std::mt19937_64 generator(rko::random_seed_seq::get_instance());

  std::generate(std::begin(data), std::end(data), gen);

  for (int i = 0; i < std::log2(sz_max / sz_begin); ++i) {
    std::shuffle(std::begin(data), std::end(data), generator);
    auto   sz_local = sz_begin * (1 << i);
    size_t nels     = nr * sz_local;
    data.resize(nels);
    out.resize(nels);
    correct.resize(nels);

    auto barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));

    auto now = ChronoClockNow();
    MPI_Alltoall(
        std::addressof(*std::begin(data)),
        sz_begin,
        MPI_INT,
        std::addressof(*std::begin(correct)),
        sz_begin,
        MPI_INT,
        MPI_COMM_WORLD);
    auto duration = ChronoClockNow() - now;

    times.push_back(std::make_pair("AlltoAll", duration));
    barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));

    factorParty(std::begin(data), std::begin(out), sz_begin, MPI_COMM_WORLD);
    duration = ChronoClockNow() - now;
    times.push_back(std::make_pair("Factor Party", duration));

    barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));

    assert(
        std::equal(std::begin(correct), std::end(correct), std::begin(out)));

    now = ChronoClockNow();

    flatFactor_selfLoops(
        std::begin(data), std::begin(out), sz_begin, MPI_COMM_WORLD);
    duration = ChronoClockNow() - now;
    times.push_back(std::make_pair("FlatFactor", duration));

    assert(
        std::equal(std::begin(correct), std::end(correct), std::begin(out)));

    barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));
    now = ChronoClockNow();
    flatHandshake(
        std::begin(data), std::begin(out), sz_begin, MPI_COMM_WORLD);
    duration = ChronoClockNow() - now;
    times.push_back(std::make_pair("Flat Handshake", duration));

    assert(
        std::equal(std::begin(correct), std::end(correct), std::begin(out)));

    barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));

    if (me == 0) {
      std::cout << "KB: " << sz_local * sizeof(value_t) / (1 << 10) << "\n";
      std::copy(
          std::begin(times),
          std::end(times),
          std::ostream_iterator<PAIR>(std::cout, "\n"));
    }
    times.clear();
    barrier = clock.Barrier(comm);
    assert(barrier.Success(comm));
  }

  MPI_Finalize();

  return 0;
}
