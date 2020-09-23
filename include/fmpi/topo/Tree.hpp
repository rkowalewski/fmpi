#pragma once

#include <fmpi/mpi/Environment.hpp>
#include <memory>

namespace fmpi {

struct Tree {
  std::vector<mpi::Rank> destinations{};
  mpi::Rank              src{};
  mpi::Rank              root{};
  uint32_t               radix{};
  Tree(mpi::Rank r, uint32_t radx)
    : root{r}
    , radix{radx} {
  }
};

/*
 * knomial:
 *
 * Description: an implementation of a k-nomial broadcast tree algorithm
 *
 * Time: (radix - 1)O(log_{radix}(comm_size))
 * Schedule length (rounds): O(log(comm_size))
 */

std::unique_ptr<Tree> knomial(
    mpi::Rank me, mpi::Rank root, uint32_t size, uint32_t radix = 2);

inline std::unique_ptr<Tree> knomial(
    const mpi::Context& ctx, mpi::Rank root, uint32_t radix = 2) {
  return knomial(ctx.rank(), root, ctx.size(), radix);
}

}  // namespace fmpi
