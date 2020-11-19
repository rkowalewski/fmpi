#include <fmpi/topo/Tree.hpp>
#include <fmpi/util/Math.hpp>
#include <fmpi/util/NumericRange.hpp>
// TLX
#include <tlx/math/ffs.hpp>
#include <tlx/math/integer_log2.hpp>
#include <tlx/math/round_to_power_of_two.hpp>

namespace fmpi {
static void knomial_tree_aux(Tree* tree, mpi::Rank me, uint32_t size) {
  /* Receive from parent */

  auto const vr    = (me - tree->root + size) % size;
  auto const radix = static_cast<int>(tree->radix);
  auto const nr    = static_cast<int>(size);
  int        mask  = 0x1;

  while (mask < nr) {
    if ((vr % (radix * mask)) != 0) {
      int parent = vr / (radix * mask) * (radix * mask);
      parent     = (parent + tree->root) % nr;
      tree->src  = mpi::Rank{parent};
      break;
    }
    mask *= radix;
  }
  mask /= radix;

  /* Send data to all children */
  while (mask > 0) {
    for (int r = 1; r < radix; r++) {
      int child = vr + mask * r;
      if (child < nr) {
        child = (child + tree->root) % nr;
        tree->destinations.push_back(mpi::Rank{child});
      }
    }
    mask /= radix;
  }
}

static void binomial_tree_aux(Tree* tree, mpi::Rank me, uint32_t size) {
  // cyclically shifted rank
  int32_t const vr = (me - tree->root + size) % size;
  auto const    nr = static_cast<int>(size);

  int d = 1;  // distance
  int r = 0;  // round
  if (vr > 0) {
    r = tlx::ffs(vr) - 1;
    d <<= r;
    auto const from = ((vr ^ d) + tree->root) % nr;
    tree->src       = mpi::Rank{from};
  } else {
    d = tlx::round_up_to_power_of_two(nr);
  }

  for (d >>= 1; d > 0; d >>= 1, ++r) {
    if (vr + d < nr) {
      auto to = (vr + d + tree->root) % nr;
      tree->destinations.push_back(mpi::Rank{to});
    }
  }
}

std::unique_ptr<Tree> knomial(
    mpi::Rank me, mpi::Rank root, uint32_t size, uint32_t radix) {
  std::unique_ptr<Tree> tree = std::make_unique<Tree>(root, radix);

  if (radix == 2) {
    binomial_tree_aux(tree.get(), me, size);
  } else {
    knomial_tree_aux(tree.get(), me, size);
  }

  return tree;
}
}  // namespace fmpi
