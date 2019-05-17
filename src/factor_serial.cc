#include <cassert>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

struct RankPair : std::pair<int, int> {
  using std::pair<int, int>::pair;
};
std::ostream& operator<<(std::ostream& os, RankPair const& p)
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

inline constexpr RankPair oneFactor_classic(int me, int r, int n)
{
  auto k_bottom = 2 * n - 1;
  return RankPair(mod(r + me, k_bottom) + 1, mod(r - me, k_bottom) + 1);
}

inline auto factorParty(int nr)
{
  assert(nr % 2 == 0);

  // We have 2n ranks
  auto n = nr / 2;

  std::vector<std::vector<RankPair>> rounds(nr);

  // Rounds
  for (int r = 1; r < nr; ++r) {
    std::vector<int> partner;
    partner.reserve(nr);
    partner[0] = r;
    partner[r] = 0;

    // generate remaining pairs
    for (int p = 1; p < n; ++p) {
      auto pair            = oneFactor_classic(p, r - 1, n);
      partner[pair.first]  = pair.second;
      partner[pair.second] = pair.first;
    }

    for (int i = 0; i < nr; ++i) {
      rounds[r].push_back(RankPair{i, partner[i]});
    }
  }

  // self loops
  for (int r = 0; r < nr; ++r) {
    rounds[0].push_back(RankPair(r, r));
  }
  return rounds;
}

inline auto flatFactor_selfLoops(int nr)
{
  std::vector<std::vector<RankPair>> res(nr);
  for (int i = 1; i <= nr; ++i) {
    for (int r = 0; r < nr; ++r) {
      res[i - 1].push_back(RankPair(r, mod(i - r, nr)));
    }
    assert(res[i - 1].size() == nr);
  }
  return res;
}

inline void flatHandshake(int nr)
{
  std::vector<std::vector<RankPair>> res(nr);
  for (int i = 1; i < nr; ++i) {
    for (int r = 0; r < nr; ++r) {
      res[i].push_back(RankPair(mod(r + i, nr), mod(r - i, nr)));
    }
    assert(res[i].size() == nr);
  }

  // self loops
  for (int r = 0; r < nr; ++r) {
    res[0].push_back(RankPair(r, r));
  }
}

inline auto hypercube(int nr)
{
  assert((nr & (nr - 1)) == 0);
  std::vector<std::vector<RankPair>> res(nr);
  for (int i = 1; i < nr; ++i) {
    for (int r = 0; r < nr; ++r) {
      res[i].push_back(RankPair(r, r ^ i));
    }
  }

  for (int r = 0; r < nr; ++r) {
    res[0].push_back(RankPair(r, r));
  }

  return res;
}

inline auto ndigits(int nr)
{
  size_t length = 1;
  while (nr /= 10) length++;
  return length;
}

void print_dot(
    std::vector<std::vector<RankPair>> tournament, std::string title)
{
  auto n_ranks = tournament.size();
  auto nd      = ndigits(n_ranks);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=LR]\n";
  std::cout << "node [style=filled shape=circle]\n";
  std::cout << "edge [arrowhead=none]\n";
  std::cout << "labelloc=t\n";
  std::cout << "label=\"" << title << "\"\n";

  for (std::size_t p = 0; p < n_ranks + 1; ++p) {
    std::cout << "subgraph p" << p << " {\n";
    std::cout << "rank=same\n";
    for (std::size_t r = 0; r < n_ranks; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      if (p == 0) {
        std::cout << " [label=\"\" xlabel=\"" << r << "\" width=.3]\n";
      } else {
        std::cout << " [label=\"\" width=.3]\n";
      }
    }
    std::cout << "{edge [style=invis]\n";

    for (std::size_t r = 0; r < n_ranks - 1; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      std::cout << "->";
    }

    std::cout << "r";
    std::cout << std::setfill('0') << std::setw(nd) << p;
    std::cout << std::setfill('0') << std::setw(nd) << n_ranks - 1;
    std::cout << "\n";

    std::cout << "}\n";
    std::cout << "}\n";
  }

  for (std::size_t p = 0; p < n_ranks; ++p) {
    for (auto const& pair : tournament[p]) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << pair.first;
      std::cout << "->r";
      std::cout << std::setfill('0') << std::setw(nd) << p + 1;
      std::cout << std::setfill('0') << std::setw(nd) << pair.second;
      std::cout << "\n";
    }
  }

  std::cout << "}\n";
}

int main(int argc, char* argv[])
{
  constexpr int n_ranks = 8;

  // std::cout << "Flat Factor excluding self loops:\n";
  auto rounds = flatFactor_selfLoops(n_ranks);
  assert(rounds.size() == n_ranks);

#if 0
  for (auto const& r : rounds) {
    std::copy(
        r.begin(), r.end(), std::ostream_iterator<RankPair>(std::cout, " "));
    std::cout << "\n";
  }
#else

  print_dot(rounds, "flat factor");

#endif

  // std::cout << "\nFlat Factor including self loops:\n";
  // flatFactor_selfLoops(n_ranks);
  // std::cout << "\nFlat Handshake excluding self loops:\n";
  // flatHandshake(n_ranks);

  return 0;
}
