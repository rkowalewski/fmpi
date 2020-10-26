#ifndef FMPI_MEMORY_THREADALLOCATOR_HPP
#define FMPI_MEMORY_THREADALLOCATOR_HPP

#include <cstddef>
#include <fmpi/detail/Assert.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>

namespace fmpi {

void* snmalloc_alloc(std::size_t n);
void  snmalloc_free(void* p, std::size_t n);

template <class T>
class ThreadAllocator {
 public:
  using value_type      = T;
  using pointer         = T*;
  using const_pointer   = const T*;
  using reference       = T&;
  using const_reference = const T&;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;
  using is_always_equal = std::is_empty<ThreadAllocator>;

  template <typename U>
  struct rebind {
    typedef ThreadAllocator<U> other;
  };

  ThreadAllocator() noexcept = default;

  template <typename U>
  ThreadAllocator(const ThreadAllocator<U>&) noexcept {
  }

  template <typename U>
  ThreadAllocator(const ThreadAllocator<U>&&) noexcept {
  }

  value_type* allocate(std::size_t n) {
    void* address = snmalloc_alloc(n * sizeof(T));
    FMPI_ASSERT(detail::isAligned(address, alignof(T)));
    return static_cast<value_type*>(address);
  }

  void deallocate(value_type* p, std::size_t size) {
    snmalloc_free(p, size * sizeof(T));
  }

  [[nodiscard]] ThreadAllocator select_on_container_copy_construction()
      const noexcept {
    return *this;
  }
};

template <class T, class U>
constexpr bool operator==(
    ThreadAllocator<T> const& /*unused*/,
    ThreadAllocator<U> const& /*unused*/) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(
    ThreadAllocator<T> const& x, ThreadAllocator<U> const& y) noexcept {
  return !(x == y);
}
}  // namespace fmpi
#endif
