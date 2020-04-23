#ifndef FMPI_MEMORY_HPP
#define FMPI_MEMORY_HPP

#include <memory>
#include <type_traits>


/// This is just copy and paste from P0211r2
/// see: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0211r2.html

#if 0

namespace fmpi {

template <class T, class A, class... Args>
auto allocator_new(A& alloc, Args&&... args) {
  using TTraits =
      typename std::allocator_traits<A>::template rebind_traits<T>;
  using TAlloc = typename std::allocator_traits<A>::template rebind_alloc<T>;

  auto a = TAlloc(alloc);
  auto p = TTraits::allocate(a, 1);

  try {
    TTraits::construct(a, to_address(p), std::forward<Args>(args)...);
    return p;
  } catch (...) {
    TTraits::deallocate(a, p, 1);
    throw;
  }
}

template <class A, class P>
void allocator_delete(A& alloc, P p) {
  using Elem = typename std::pointer_traits<P>::element_type;
  using Traits =
      typename std::allocator_traits<A>::template rebind_traits<Elem>;

  Traits::destroy(alloc, to_address(p));
  Traits::deallocate(alloc, p, 1);
}

template <class A>
struct allocation_deleter {
  using pointer = typename std::allocator_traits<A>::pointer;

  A a_;  // exposition only

  allocation_deleter(const A& a) noexcept
    : a_(a) {
  }

  void operator()(pointer p) {
    allocator_delete(a_, p);
  }
};

template <class T, class A, class... Args>
auto allocate_unique(A& alloc, Args&&... args) {
  using TAlloc = typename std::allocator_traits<A>::template rebind_alloc<T>;
  return std::unique_ptr<T, allocation_deleter<TAlloc>>(
      allocator_new<T>(alloc, std::forward<Args>(args)...), alloc);
}

}  // namespace fmpi

#endif

#endif
