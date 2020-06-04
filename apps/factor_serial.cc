#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <fmpi/Math.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Schedule.hpp>

using RankPair = std::pair<mpi::Rank, mpi::Rank>;

inline auto oneFactor(int nr) {
  std::vector<std::vector<RankPair>> res(nr);

  for (auto&& me : fmpi::range(mpi::Rank{0}, mpi::Rank{nr})) {
    fmpi::OneFactor schedule{static_cast<uint32_t>(nr), me};
    for (auto&& phase : fmpi::range(schedule.phaseCount())) {
      res[phase].emplace_back(me, schedule.sendRank(phase));
    }
  }
  if (nr % 2 == 0) {
    // self loops
    for (auto&& r : fmpi::range(nr)) {
      res[nr - 1].emplace_back(r, r);
    }
  }

  return res;
}

inline auto flatHandshake(int nr) {
  std::vector<std::vector<RankPair>> res(nr);

  for (auto&& me : fmpi::range(mpi::Rank{0}, mpi::Rank{nr})) {
    fmpi::FlatHandshake schedule{static_cast<uint32_t>(nr), me};
    for (auto&& phase : fmpi::range(schedule.phaseCount())) {
      res[phase].emplace_back(
          schedule.sendRank(phase), schedule.recvRank(phase));
    }
  }
  return res;
}

inline auto ndigits(int nr) {
  size_t length = 1;
  while ((nr /= 10) != 0) {
    length++;
  }
  return length;
}

void print_dot(
    std::vector<std::vector<RankPair>> tournament, const std::string& title) {
  (void)title;
  auto n_ranks = tournament.size();
  auto nd      = ndigits(n_ranks);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=TB]\n";
  std::cout << "node [style=filled shape=circle]\n";
  std::cout << "edge [arrowhead=none]\n";
  // std::cout << "labelloc=t\n";
  // std::cout << "label=\"" << title << "\"\n";

  for (std::size_t p = 0; p < n_ranks + 1; ++p) {
    std::cout << "subgraph p" << p << " {\n";
    std::cout << "rank=same\n";
    for (std::size_t r = 0; r < n_ranks; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      if (p == 0) {
        std::cout << R"( [label="" xlabel=")" << r << "\" width=.3]\n";
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
        // std::cout << "[style=invis]";
      }
      std::cout << "\n";
    }
  }

  std::cout << "}\n";
}

void print_dot_directed(
    std::vector<std::vector<RankPair>> tournament, const std::string& title) {
  (void)title;
  auto n_ranks = tournament.size();
  auto nd      = ndigits(n_ranks);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=TB]\n";
  std::cout << "node [style=filled shape=circle]\n";
  std::cout << "edge [arrowsize=0.5]\n";
  // std::cout << "labelloc=t\n";
  // std::cout << "label=\"" << title << "\"\n";

  for (std::size_t p = 0; p < n_ranks + 1; ++p) {
    std::cout << "subgraph p" << p << " {\n";
    std::cout << "rank=same\n";
    for (std::size_t r = 0; r < n_ranks; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      if (p == 0) {
        std::cout << R"( [label="" xlabel=")" << r << "\" width=.3]\n";
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
        std::cout << "[arrowhead=none]";
        std::cout << "\n";
      } else {
        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.first;
        std::cout << "[color=cornflowerblue";

        // if (p > 3) {
        // std::cout << ",style=invis]";
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
          // std::cout << "[style=invis]";
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
    2>
    algos = {std::make_pair("oneFactor", oneFactor),
             std::make_pair("ring", flatHandshake)};

void print_usage(const std::string& prog) {
  std::cout << "usage: " << prog << "<algorithm> <nprocs>\n\n";

  std::cout << "supported algorithms: ";
  for (auto& algo : algos) {
    std::cout << algo.first << ", ";
  }
  std::cout << "\n";
}
int main(int argc, char* argv[]) {
  if (argc < 3) {
    print_usage(std::string(argv[0]));
    return 1;
  }

  std::string algo    = argv[1];
  auto        n_ranks = std::atoi(argv[2]);

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

  auto& name = (*it).first;
  if (name == "ring") {
    print_dot_directed(rounds, name);
  } else {
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

  return 0;
}
