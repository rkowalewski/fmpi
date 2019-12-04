#ifndef FMPI_MEMORY_H
#define FMPI_MEMORY_H

#include <algorithm>
#include <list>
#include <tlx/stack_allocator.hpp>
#include <vector>

namespace fmpi {

static constexpr std::size_t MAX_STACK_SIZE_BUF = 1024;

template <class T, std::size_t N>
struct SmallVector {
 private:
  static constexpr std::size_t nbytes =
      (MAX_STACK_SIZE_BUF < N) ? MAX_STACK_SIZE_BUF : N;

 public:
  using arena     = tlx::StackArena<nbytes>;
  using allocator = tlx::StackAllocator<T, nbytes>;
  using vector    = std::vector<T, allocator>;
};

}  // namespace fmpi
#endif
