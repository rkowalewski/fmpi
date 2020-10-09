#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmpi/Schedule.hpp>
#include <fmpi/util/Math.hpp>
#include <fmpi/util/NumericRange.hpp>
//#include <fmpi/topo/BinaryTree.hpp>
#include <fmpi/topo/Tree.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <tlx/math/ffs.hpp>
#include <tlx/math/integer_log2.hpp>
#include <tlx/math/round_to_power_of_two.hpp>
#include <utility>
#include <vector>

using RankPair = std::pair<mpi::Rank, mpi::Rank>;

//////////////////////////////////
/////// Alltoall /////////////////
//////////////////////////////////

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

inline auto bruck(int nr) {
  auto const nway = 1;
  auto const base = nway + 1;

  auto const phase_count = std::ceil(fmpi::log(base, nr));

  std::vector<std::vector<RankPair>> res(phase_count);

  for (auto&& me : fmpi::range(mpi::Rank{0}, mpi::Rank{nr})) {
    for (auto&& r : fmpi::range(phase_count)) {
      for (auto&& w : fmpi::range(1, base)) {
        auto const j = static_cast<mpi::Rank>(w * std::pow(base, r));
        res[r].emplace_back(
            fmpi::mod(me + j, static_cast<mpi::Rank>(nr)),
            fmpi::mod(me - j, static_cast<mpi::Rank>(nr)));
      }
    }
  }
  return res;
}

//////////////////////////////////
/////// Binomial ////////////////
//////////////////////////////////

#if 0

/* better binomial bcast
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
 *
 * Algorithm:
 * - each node with vrank > 2^r and vrank < 2^r+1 receives from node
 *   vrank - 2^r (vrank=1 receives from 0, vrank 0 receives never)
 * - each node sends each round r to node vrank + 2^r
 * - a node stops to send if 2^r > commsize
 */
#define RANK2VRANK(rank, vrank, root) \
  {                                   \
    vrank = rank;                     \
    if (rank == 0) vrank = root;      \
    if (rank == root) vrank = 0;      \
  }
#define VRANK2RANK(rank, vrank, root) \
  {                                   \
    rank = vrank;                     \
    if (vrank == 0) rank = root;      \
    if (vrank == root) rank = 0;      \
  }

inline fmpi::Tree binomial_tree_aux2(int me, int nr, int root) {
  int maxr, vrank, peer;

  maxr = std::ceil(fmpi::log(2, nr));

  auto tree = fmpi::Tree{me, root, 2};

  RANK2VRANK(me, vrank, root);

  /* receive from the right hosts  */
  if (vrank != 0) {
    for (int r = 0; r < maxr; ++r) {
      if ((vrank >= (1 << r)) && (vrank < (1 << (r + 1)))) {
        VRANK2RANK(peer, vrank - (1 << r), root);
        tree.src = mpi::Rank{peer};
        // std::cout << "[r " << me << ", vr " << vrank
        //          << "] recv from: " << peer << ", round: " << r << "\n";
      }
    }
  }

  /* now send to the right hosts */
  for (int r = 0; r < maxr; ++r) {
    if (((vrank + (1 << r) < nr) && (vrank < (1 << r))) || (vrank == 0)) {
      VRANK2RANK(peer, vrank + (1 << r), root);
      // std::cout << "[r " << me << ", vr " << vrank << "] send to: " << peer
      //          << ", round: " << r << "\n";
      tree.destinations.push_back(mpi::Rank{peer});
    }
  }

  return tree;
}
#endif

//////////////////////////////////
/////// K-nomial /////////////////
//////////////////////////////////

// Useful only for bcast
inline auto knomial_tree(int nr, int root, int radix = 2) {
  std::vector<std::vector<RankPair>> res(nr);

  for (auto&& me :
       fmpi::range(static_cast<mpi::Rank>(0), static_cast<mpi::Rank>(nr))) {
    auto tree = fmpi::knomial(me, static_cast<mpi::Rank>(root), nr, radix);
    for (auto&& r : tree->destinations) {
      res[me].emplace_back(me, r);
    }
  }

  return res;
}

//////////////////////////////////
/////// Regular Tree /////////////
//////////////////////////////////
#if 0

static int pown(int fanout, int num) {
  int j, p = 1;
  if (num < 0) return 0;
  if (1 == num) return fanout;
  if (2 == fanout) {
    return p << num;
  } else {
    for (j = 0; j < num; j++) {
      p *= fanout;
    }
  }
  return p;
}

static int calculate_level(int fanout, int rank) {
  int level, num;
  if (rank < 0) return -1;
  for (level = 0, num = 0; num <= rank; level++) {
    num += pown(fanout, level);
  }
  return level - 1;
}

static int calculate_num_nodes_up_to_level(int fanout, int level) {
  /* just use geometric progression formula for sum:
     a^0+a^1+...a^(n-1) = (a^n-1)/(a-1) */
  return ((pown(fanout, level) - 1) / (fanout - 1));
}

inline auto regular_tree_aux(int me, int nr, int root, uint32_t radix) {
  if (radix < 1) {
  }
  if (radix > 32) {
  }

  /*
   * Get size and rank of the process in this communicator
   */
  auto tree = fmpi::Tree{me, root, radix};

  /*
   * Initialize tree
   */
  /* return if we have less than 2 processes */
  if (nr < 2) {
    return tree;
  }

  /*
   * Shift all ranks by root, so that the algorithm can be
   * designed as if root would be always 0
   * shiftedrank should be used in calculating distances
   * and position in tree
   */
  auto const vr = fmpi::mod(me - root, nr);

  /* calculate my level */
  auto level = calculate_level(radix, vr);
  auto delta = pown(radix, level);

  /* find my children */
  for (auto i = 0; i < tree.radix; ++i) {
    auto schild = vr + delta * (i + 1);
    if (schild < nr) {
      tree.destinations.push_back(
          static_cast<mpi::Rank>((schild + root) % nr));
    } else {
      break;
    }
  }

  /* find my parent */

  /* total number of nodes on levels above me */
  auto const slimit  = calculate_num_nodes_up_to_level(radix, level);
  auto       sparent = vr;
  if (sparent < radix) {
    sparent = 0;
  } else {
    while (sparent >= slimit) {
      sparent -= delta / radix;
    }
  }
  tree.src = mpi::Rank{(sparent + root) % nr};

  return tree;
}

inline auto regular_tree(int nr, int root, int fanout = 2) {
  std::vector<std::vector<RankPair>> res(nr);

  for (auto me : fmpi::range(nr)) {
    auto tree = regular_tree_aux(me, nr, root, fanout);
    for (auto&& r : tree.destinations) {
      res[me].emplace_back(me, r);
    }
  }

  return res;
}

auto pipeline_aux(int me, int nr, int root, uint32_t radix) {
  constexpr uint32_t max_fanout = 32;

  radix = std::min(
      {std::max(radix, 1u), max_fanout, static_cast<uint32_t>(nr - 1)});

  /*
   * Allocate space for topology arrays if needed
   */
  auto tree = fmpi::Tree{me, root, static_cast<uint32_t>(radix)};

  /*
   * Set root & numchain
   */
  tree.src = mpi::Rank{root};

  /*
   * Shift ranks
   */
  std::cout << "test\n" << std::endl;
  auto const vr = fmpi::mod(me - root, nr);

  /*
   * Special case - fanout == 1
   */
  if (tree.radix == 1) {
    if (vr == 0) {
      // chain->tree_prev = -1
    } else {
      tree.src = static_cast<mpi::Rank>((vr - 1 + root) % nr);
    }

    if ((vr + 1) >= nr) {
      // chain->tree_next[0] = -1;
      // chain->tree_nextsize = 0;
    } else {
      tree.destinations.push_back(
          static_cast<mpi::Rank>((vr + 1 + root) % nr));
    }
    return tree;
  }

  /* Let's handle the case where there is just one node in the communicator */
  if (nr == 1) {
    return tree;
  }

  /*
   * Calculate maximum chain length
   */
  int maxchainlen = (nr - 1) / tree.radix;
  int mark;
  if ((nr - 1) % tree.radix != 0) {
    maxchainlen++;
    mark = (nr - 1) % tree.radix;
  } else {
    mark = tree.radix + 1;
  }

  /*
   * Find your own place in the list of shifted ranks
   */
  if (vr != 0) {
    int column, head, len;
    if (vr - 1 < (mark * maxchainlen)) {
      column = (vr - 1) / maxchainlen;
      head   = 1 + column * maxchainlen;
      len    = maxchainlen;
    } else {
      column = mark + (vr - 1 - mark * maxchainlen) / (maxchainlen - 1);
      head   = mark * maxchainlen + 1 + (column - mark) * (maxchainlen - 1);
      len    = maxchainlen - 1;
    }

    if (vr == head) {
      tree.src = mpi::Rank{0}; /*root*/
    } else {
      tree.src = mpi::Rank{vr - 1}; /* me -1 */
    }
    if (vr == (head + len - 1)) {
      // chain->tree_next[0]  = -1;
      // chain->tree_nextsize = 0;
    } else {
      if ((vr + 1) < nr) {
        tree.destinations.push_back(static_cast<mpi::Rank>(vr + 1));
      } else {
        // chain->tree_next[0]  = -1;
        // chain->tree_nextsize = 0;
      }
    }

    tree.src = static_cast<mpi::Rank>((tree.src + root) % nr);
    if (tree.destinations.size() &&
        tree.destinations.front() != mpi::Rank::null()) {
      auto& fstChild = tree.destinations.front();
      fstChild       = static_cast<mpi::Rank>((fstChild + root) % nr);
    }
  } else {
    /*
     * Unshift values
     */
    tree.src = mpi::Rank::null();
    tree.destinations.push_back(static_cast<mpi::Rank>((root + 1) % nr));
    for (auto i = 1; i < tree.radix; i++) {
      tree.destinations.push_back(mpi::Rank{maxchainlen});
      auto& back = tree.destinations.back();
      if (i > mark) {
        back--;
      }
      back = back % mpi::Rank{nr};
    }
    // chain->tree_nextsize = tree.radix;
    // assert(tree.destinations.size()
  }

  return tree;
}

inline auto pipeline(int nr, int root, int fanout = 2) {
  std::vector<std::vector<RankPair>> res(nr);

  for (auto me : fmpi::range(nr)) {
    auto tree = pipeline_aux(me, nr, root, fanout);
    for (auto&& r : tree.destinations) {
      res[me].emplace_back(me, r);
    }
  }

  return res;
}

inline auto binary_tree(int nr, int root) {
  int left, right, parent;

  std::vector<std::vector<RankPair>> res(nr);
  for (auto me : fmpi::range(nr)) {
    fmpi::Create(me, nr, &left, &right, &parent);
    // auto tree = Tree{me, root, 2};
    // tree.destinations.push_back(mpi::Rank{left});
    // tree.destinations.push_back(mpi::Rank{right});
    // tree.src = mpi::Rank{parent};
    if (left != -1) res[me].emplace_back(me, left);
    if (right != -1) res[me].emplace_back(me, right);
  }
  return res;
}
ompi_coll_tree_t*
ompi_coll_base_topo_build_in_order_bintree( struct ompi_communicator_t* comm )
{
    int rank, size, myrank, rightsize, delta, parent, lchild, rchild;
    ompi_coll_tree_t* tree;

    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    tree = (ompi_coll_tree_t*)malloc(COLL_TREE_SIZE(MAXTREEFANOUT));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }

    tree->tree_root     = MPI_UNDEFINED;
    tree->tree_nextsize = MPI_UNDEFINED;

    /*
     * Initialize tree
     */
    tree->tree_fanout   = 2;
    tree->tree_bmtree   = 0;
    tree->tree_root     = size - 1;
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    tree->tree_next[0]  = -1;
    tree->tree_next[1]  = -1;
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:topo_build_in_order_tree Building fo %d rt %d",
                 tree->tree_fanout, tree->tree_root));

    /*
     * Build the tree
     */
    myrank = rank;
    parent = size - 1;
    delta = 0;

    while ( 1 ) {
        /* Compute the size of the right subtree */
        rightsize = size >> 1;

        /* Determine the left and right child of this parent  */
        lchild = -1;
        rchild = -1;
        if (size - 1 > 0) {
            lchild = parent - 1;
            if (lchild > 0) {
                rchild = rightsize - 1;
            }
        }

        /* The following cases are possible: myrank can be
           - a parent,
           - belong to the left subtree, or
           - belong to the right subtee
           Each of the cases need to be handled differently.
        */

        if (myrank == parent) {
            /* I am the parent:
               - compute real ranks of my children, and exit the loop. */
            if (lchild >= 0) tree->tree_next[0] = lchild + delta;
            if (rchild >= 0) tree->tree_next[1] = rchild + delta;
            break;
        }
        if (myrank > rchild) {
            /* I belong to the left subtree:
               - If I am the left child, compute real rank of my parent
               - Iterate down through tree:
               compute new size, shift ranks down, and update delta.
            */
            if (myrank == lchild) {
                tree->tree_prev = parent + delta;
            }
            size = size - rightsize - 1;
            delta = delta + rightsize;
            myrank = myrank - rightsize;
            parent = size - 1;

        } else {
            /* I belong to the right subtree:
               - If I am the right child, compute real rank of my parent
               - Iterate down through tree:
               compute new size and parent,
               but the delta and rank do not need to change.
            */
            if (myrank == rchild) {
                tree->tree_prev = parent + delta;
            }
            size = rightsize;
            parent = rchild;
        }
    }

    if (tree->tree_next[0] >= 0) { tree->tree_nextsize = 1; }
    if (tree->tree_next[1] >= 0) { tree->tree_nextsize += 1; }

    return tree;
}
#endif

inline auto ndigits(int nr) {
  size_t length = 1;
  while ((nr /= 10) != 0) {
    length++;
  }
  return length;
}

void print_dot(
    std::vector<std::vector<RankPair>> tournament, const std::string& title);

void print_dot_directed(
    std::vector<std::vector<RankPair>> tournament,
    const std::size_t&                 nodes,
    const std::string&                 title);

void print_dot_tree(std::vector<std::vector<RankPair>> pairs);

static std::array<
    std::pair<
        std::string,
        std::function<std::vector<std::vector<RankPair>>(int)>>,
    3>
    algos = {
        std::make_pair("oneFactor", oneFactor),
        std::make_pair("ring", flatHandshake),
        std::make_pair("dissemination", bruck)};

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

  std::string algo = argv[1];
  auto        nr   = std::atoi(argv[2]);

  //  binomial_tree(nr, 0);
  //
  //  return 0;

  if (algo != "tree") {
    auto* it = std::find_if(
        std::begin(algos), std::end(algos), [&algo](auto const& pair) {
          return pair.first == algo;
        });

    if (it == std::end(algos)) {
      std::cout << "invalid algorithm\n";
      return 1;
    }

    auto rounds = (*it).second(nr);

    auto& name = it->first;
    if (name == "ring" || name == "dissemination") {
      print_dot_directed(rounds, nr, name);
    } else {
      print_dot(rounds, name);
    }

    std::cout << "\n";
  } else {
    constexpr int root = 0;
    print_dot_tree(knomial_tree(nr, root, 3));
    // print_dot_tree(regular_tree(nr, root));
    // print_dot_tree(knomial_tree(nr, root, 3));
    // print_dot_tree(binary_tree(nr, root));
  }

  return 0;
}

void print_dot(
    std::vector<std::vector<RankPair>> tournament, const std::string& title) {
  (void)title;
  auto nr = tournament.size();
  auto nd = ndigits(nr);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=TB ranksep=1 nodesep=1]\n";
  std::cout << "node [style=filled shape=circle]\n";
  std::cout << "edge [arrowhead=none]\n";
  // std::cout << "labelloc=t\n";
  // std::cout << "label=\"" << title << "\"\n";

  for (std::size_t p = 0; p < nr + 1; ++p) {
    std::cout << "subgraph p" << p << " {\n";
    std::cout << "rank=same\n";
    for (std::size_t r = 0; r < nr; ++r) {
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

    for (std::size_t r = 0; r < nr - 1; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      std::cout << "->";
    }

    std::cout << "r";
    std::cout << std::setfill('0') << std::setw(nd) << p;
    std::cout << std::setfill('0') << std::setw(nd) << nr - 1;
    std::cout << "\n";

    std::cout << "}\n";
    std::cout << "}\n";
  }

  for (std::size_t r = 0; r < nr; ++r) {
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
    std::vector<std::vector<RankPair>> tournament,
    const std::size_t&                 nodes,
    const std::string&                 title) {
  (void)title;
  auto nd = ndigits(nodes);
  std::cout << "digraph G {\n";
  std::cout << "graph [rankdir=TB ranksep=1 nodesep=1]\n";
  std::cout << "node [shape=circle]\n";
  // std::cout << "edge [arrowsize=0.5]\n";
  // std::cout << "labelloc=t\n";
  // std::cout << "label=\"" << title << "\"\n";

  for (std::size_t p = 0; p < tournament.size() + 1; ++p) {
    std::cout << "subgraph p" << p << " {\n";
    std::cout << "rank=same\n";
    for (std::size_t r = 0; r < nodes; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      std::cout << " [label=\"" << r << "\" width=.3]\n";
      if (p == 0) {
        // std::cout << R"( [label="" xlabel=")" << r << "\" width=.3]\n";
      } else {
      }
    }

    std::cout << "{edge [style=invis]\n";

    for (std::size_t r = 0; r < nodes - 1; ++r) {
      std::cout << "r";
      std::cout << std::setfill('0') << std::setw(nd) << p;
      std::cout << std::setfill('0') << std::setw(nd) << r;
      std::cout << "->";
    }

    std::cout << "r";
    std::cout << std::setfill('0') << std::setw(nd) << p;
    std::cout << std::setfill('0') << std::setw(nd) << nodes - 1;
    std::cout << "\n";

    std::cout << "}\n";
    std::cout << "}\n";
  }

  for (std::size_t r = 0; r < tournament.size(); ++r) {
    for (unsigned p = 0; p < tournament[r].size(); ++p) {
      auto const& partners = tournament[r][p];
      if (partners.first == partners.second) {
        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.first;
        // std::cout << "[arrowhead=none";

        // if (!(p % 2)) {
        //  std::cout << ",style=invis]";
        //} else {
        // std::cout << "]";
        //}
        std::cout << "\n";

        std::cout << "\n";
      } else {
        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.first;
        std::cout << "[color=black";

        // if (!(p % 2)) {
        //  std::cout << ",style=invis]";
        //} else {
        std::cout << "]";
        //}
        std::cout << "\n";

        std::cout << "r";
        std::cout << std::setfill('0') << std::setw(nd) << r + 1;
        std::cout << std::setfill('0') << std::setw(nd) << partners.second;
        std::cout << "->r";
        std::cout << std::setfill('0') << std::setw(nd) << r;
        std::cout << std::setfill('0') << std::setw(nd) << p;
        // if (!(p % 2)) {
        std::cout << "[style=invis]";
        //}
        std::cout << "\n";
      }
    }
  }

  std::cout << "}\n";
}

void print_dot_tree(std::vector<std::vector<RankPair>> pairs) {
  std::cout << "digraph BST {\n";
  std::cout << "node [shap=\"circle\"];\n";

  for (auto&& r : fmpi::range(pairs.size())) {
    auto partners = pairs[r];
    for (auto&& r : partners) {
      std::cout << "p_" << r.first << " -> p_" << r.second << ";\n";
    }
  }

  std::cout << "}\n";
}

#if 0
struct CollTree {
  int32_t tree_root;
  int32_t tree_fanout;
  int32_t tree_bmtree;
  int32_t tree_prev;
  int32_t tree_nextsize;
  int32_t tree_next[];
};

#endif
