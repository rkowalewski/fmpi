#ifndef FMPI__MEMORY_H
#define FMPI__MEMORY_H

#include <vector>

#include <tlx/stack_allocator.hpp>

namespace fmpi {

template <std::size_t N>
using stack_arena = tlx::StackArena<N>;

template <class T, std::size_t N>
using stack_allocator = tlx::StackAllocator<T, N>;

template <class T, std::size_t StackSize>
using small_vector = std::vector<T, tlx::StackAllocator<T, StackSize>>;

}  // namespace fmpi
#endif
