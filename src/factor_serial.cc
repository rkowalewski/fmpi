#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
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

inline auto oneFactor_odd(int nr)
{
  assert(nr % 2);

  std::vector<std::vector<RankPair>> res(nr);
  for (int me = 1; me <= nr; ++me) {
    for (int r = 0; r < nr; ++r) {
      res[me - 1].push_back(RankPair(r, mod(me - r, nr)));
    }
    assert(res[me - 1].size() == std::size_t(nr));
  }
  return res;
}

inline auto oneFactor_even(int nr)
{
  assert(nr % 2 == 0);

  std::vector<std::vector<RankPair>> res(nr);

  for (int me = 0; me < nr; ++me) {
    for (int r = 0; r < nr - 1; ++r) {
      auto idle = mod(nr * r / 2, nr - 1);
      if (me == nr - 1) {
        res[r + 1].push_back(RankPair{me, idle});
      }
      else if (me == idle) {
        res[r + 1].push_back(RankPair{idle, nr - 1});
      }
      else {
        res[r + 1].push_back(RankPair{me, mod(r - me, nr - 1)});
      }
    }
  }

  // self loops
  for (int r = 0; r < nr; ++r) {
    res[0].push_back(RankPair(r, r));
  }

  return res;
}

inline auto oneFactor(int nr)
{
  if ((nr % 2) != 0) {
    return oneFactor_odd(nr);
  }
  return oneFactor_even(nr);
}

inline auto flatHandshake(int nr)
{
  std::vector<std::vector<RankPair>> res(nr);
  for (int i = 1; i < nr; ++i) {
    for (int r = 0; r < nr; ++r) {
      res[i].push_back(RankPair(mod(r + i, nr), mod(r - i, nr)));
    }
    assert(res[i].size() == std::size_t(nr));
  }

  // self loops
  for (int r = 0; r < nr; ++r) {
    res[0].push_back(RankPair(r, r));
  }
  return res;
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
  while ((nr /= 10) != 0) { length++;
}
  return length;
}

void print_dot(
    std::vector<std::vector<RankPair>> tournament, const std::string& title)
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
        std::cout << R"( [label="" xlabel=")" << r << "\" width=.3]\n";
      }
      else {
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

  for (std::size_t r = 0; r < n_ranks; ++r) {
    for (std::size_t p = 0; p < tournament.size(); ++p) {
      auto const& pair = tournament[r][p];
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << r;
      std::cout << std::setfill('0') << std::setw(nd) << pair.first;
      std::cout << "->r";
      std::cout << std::setfill('0') << std::setw(nd) << r + 1;
      std::cout << std::setfill('0') << std::setw(nd) << pair.second;
      if (p > 3) {
        //std::cout << "[style=invis]";
      }
      std::cout << "\n";
    }
  }

  std::cout << "}\n";
}

void print_dot_directed(
    std::vector<std::vector<RankPair>> tournament, const std::string& title)
{
  auto n_ranks = tournament.size();
  auto nd      = ndigits(n_ranks);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=LR]\n";
  std::cout << "node [style=filled shape=circle]\n";
  std::cout << "edge [arrowsize=0.5]\n";
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
        std::cout << R"( [label="" xlabel=")" << r << "\" width=.3]\n";
      }
      else {
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

  for (std::size_t r = 0; r < n_ranks; ++r) {
    for (unsigned p = 0; p < tournament[r].size(); ++p) {
      auto const& partners = tournament[r][p];
      if (partners.first == partners.second) {
        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.first;
        std::cout << "[color=cornflowerblue, arrowhead=none]";
        std::cout << "\n";
      }
      else {
        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.first;
        std::cout << "[color=cornflowerblue";

       // if (p > 3) {
          std::cout << ",style=invis]";
       // }
       // else {
          std::cout << "]";
       // }
        std::cout << "\n";

        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.second;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        if (p > 3) {
          //std::cout << "[style=invis]";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "}\n";
}

static std::array<
    std::pair<
        std::string,
        std::function<std::vector<std::vector<RankPair>>(int)>>,
    3>
    algos = {std::make_pair("oneFactor", oneFactor),
             std::make_pair("hypercube", hypercube),
             std::make_pair("personalized_exchange", flatHandshake)};

void print_usage(const std::string& prog)
{
  std::cout << "usage: " << prog << "<algorithm> <nprocs>\n\n";

  std::cout << "supported algorithms: ";
  for (auto& algo : algos) {
    std::cout << algo.first << ", ";
  }
  std::cout << "\n";
}
int main(int argc, char* argv[])
{
  if (argc < 3) {
    print_usage(std::string(argv[0]));
    return 1;
  }

  std::string algo    = argv[1];
  auto        n_ranks = std::atoi(argv[2]);

  auto isPower2 = (n_ranks & (n_ranks - 1)) == 0;
  if (algo == "hypercube" && !isPower2) {
    std::cout << "hypercube works only with power of 2\n";
    return 1;
  }

  auto it = std::find_if(
      std::begin(algos), std::end(algos), [&algo](auto const& pair) {
        return pair.first == algo;
      });

  if (it == std::end(algos)) {
    std::cout << "invalid algorithm\n";
    return 1;
  }

  // std::cout << "Flat Factor excluding self loops:\n";
  auto rounds = (*it).second(n_ranks);
  assert(rounds.size() == std::size_t(n_ranks));

#if 0
  for (auto const& r : rounds) {
    std::copy(
        r.begin(), r.end(), std::ostream_iterator<RankPair>(std::cout, " "));
    std::cout << "\n";
  }
#else

  auto& name = (*it).first;
  if (name == "personalized_exchange") {
    print_dot_directed(rounds, name);
  }
  else {
    print_dot(rounds, name);
  }

  std::cout << "\n";

#if 0
  auto size = n_ranks;
  for (int rank = 0; rank < size; ++rank) {
    const int target = (rank + 1) % size;
    const int source = (rank - 1 + size) % size;

    std::cout << rank << " (" << target << ", " << source << "): ";

    for (int it = 0; it != size - 1; ++it) {
      int recv_pe = (size + rank - it - 1) % size;
      int send_pe = (size + rank - it) % size;
      std::cout << "(" << recv_pe << ", " << send_pe << "), ";
    }
    std::cout << "\n";
  }

  int max_height = std::ceil(std::log2(size));
  if (std::pow(2, max_height) < size) {
    max_height++;
  }

  std::vector<std::vector<RankPair>> tree;
  tree.resize(max_height + 1);

  std::cout << "constructing the tree\n";
  auto root = 3;
  for (int rank = 0; rank < size; ++rank) {
    int new_rank   = (rank - root + size) % size;
    int own_height = 0;
    for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
      own_height++;

    tree.at(own_height).push_back(RankPair(rank, new_rank));
  }

  for (std::size_t idx = 0; idx < tree.size(); ++idx) {
    auto const& level = tree.at(idx);
    std::cout << "level " << idx << ": ";
    std::copy(
        level.begin(),
        level.end(),
        std::ostream_iterator<RankPair>(std::cout, " "));
    std::cout << "\n";
  }
  std::cout << "traversing the tree\n";
  for (int rank = 0; rank < size; ++rank) {
    int new_rank   = (rank - root + size) % size;
    int own_height = 0;
    for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
      own_height++;
    // std::cout << rank << " own_height " << own_height << "\n";

    int height = 1;
    while (height <= own_height) {
      int tmp_src = new_rank + std::pow(2, height - 1);
      // std::cout << rank << " tmp_src: " << tmp_src << "\n";
      if (tmp_src < size) {
        int src = (tmp_src + root) % size;
        std::cout << rank << " <- " << RankPair(tmp_src, src) << "\n";
        height++;
      }
      else {
        // Source rank larger than comm size
        height++;
      }
    }
    // std::cout << rank << ", height: " << height << "\n";
    if (rank != root) {
      int tmp_dest = new_rank - std::pow(2, height - 1);
      int dest     = (tmp_dest + root) % size;

      std::cout << rank << " -> " << RankPair(tmp_dest, dest) << "\n";
    }
  }

#endif
#endif

  return 0;
}
