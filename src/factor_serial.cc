#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

using rank_pair = std::pair<int, int>;

std::ostream& operator<<(std::ostream& os, rank_pair const& p)
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

inline void factorParty(int nr)
{
  assert(nr % 2 == 0);

  // We have 2n ranks
  auto n = nr / 2;

  std::vector<int> partner;
  partner.reserve(nr);

  // Rounds
  for (int r = 1; r < nr; ++r) {
    std::cout << "F" << r << " = { ";
    partner[0] = r;
    partner[r] = 0;

    // generate remaining pairs
    for (int p = 1; p < n; ++p) {
      auto pair            = oneFactor_classic(p, r - 1, n);
      partner[pair.first]  = pair.second;
      partner[pair.second] = pair.first;
    }

    for (int i = 0; i < nr; ++i) {
      std::cout << rank_pair{i, partner[i]} << ", ";
    }

    std::cout << "}\n";
  }

  std::cout << "F" << nr << " = { ";

  // self loops
  for (int r = 0; r < nr; ++r) {
    auto pair = std::make_pair(r, r);
    std::cout << pair << ", ";
  }
  std::cout << "}\n";
}

inline void flatFactor_selfLoops(int nr)
{
  for (int i = 1; i <= nr; ++i) {
    std::cout << "F" << i << " = { ";
    for (int r = 0; r < nr; ++r) {
      std::cout << rank_pair(r, mod(i - r, nr)) << ", ";
    }
    std::cout << "}\n";
  }
}

inline void flatHandshake(int nr)
{
  for (int i = 1; i < nr; ++i) {
    std::cout << "F" << i + 1 << " = { ";
    for (int r = 0; r < nr; ++r) {
      auto pair = std::make_pair(mod(r + i, nr), mod(r - i, nr));
      std::cout << pair << ", ";
    }
    std::cout << "}\n";
  }
  std::cout << "F" << nr << " = { ";

  //self loops
  for (int r = 0; r < nr; ++r) {
    auto pair = std::make_pair(r, r);
    std::cout << pair << ", ";
  }
  std::cout << "}\n";
}

int main(int argc, char* argv[])
{
  constexpr int n_ranks = 16;

  std::cout << "Flat Factor excluding self loops:\n";
  factorParty(n_ranks);
  std::cout << "\nFlat Factor including self loops:\n";
  flatFactor_selfLoops(n_ranks);
  std::cout << "\nFlat Handshake excluding self loops:\n";
  flatHandshake(n_ranks);

  return 0;
}
