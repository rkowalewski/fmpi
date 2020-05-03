#include <fmpi/memory/ThreadAllocator.hpp>

#include <snmalloc.h>

void* fmpi::snmalloc_alloc(std::size_t n) {
  auto* allocator = snmalloc::ThreadAlloc::get();
  return allocator->alloc(n);
}

void fmpi::snmalloc_free(void* p, std::size_t n) {
  auto* allocator = snmalloc::ThreadAlloc::get();
  allocator->dealloc(p, n);
}
