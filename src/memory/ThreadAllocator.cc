#include <fmpi/memory/ThreadAllocator.hpp>

#ifdef __INTEL_COMPILER
#pragma warning(disable: 2196)  // warning #2196: routine is both "inline" and "noinline"
#endif

#include <snmalloc.h>

void* fmpi::snmalloc_alloc(std::size_t n) {
  auto* allocator = snmalloc::ThreadAlloc::get();
  return allocator->alloc(n);
}

void fmpi::snmalloc_free(void* p, std::size_t n) {
  auto* allocator = snmalloc::ThreadAlloc::get();
  allocator->dealloc(p, n);
}
