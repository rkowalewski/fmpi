#ifndef FMPI_MEMORY_STDALLOCATOR_HPP
#define FMPI_MEMORY_STDALLOCATOR_HPP

#include <cstddef>

#include <fmpi/Debug.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>

namespace fmpi {

void* snmalloc_alloc(std::size_t n);
void  snmalloc_free(void* p, std::size_t n);

template <class T>
class ThreadAllocator {
  static_assert(
      std::is_trivially_copyable<T>::value,
      "MPI always requires trivially copyable types");

 public:
  using value_type = T;

  value_type* allocate(std::size_t n) {
    void* address = snmalloc_alloc(n * sizeof(T));
    FMPI_ASSERT(detail::isAligned(address, alignof(T)));
    return static_cast<value_type*>(address);
  }

  void deallocate(value_type* p, std::size_t size) {
    snmalloc_free(p, size * sizeof(T));
  }

  [[nodiscard]] ThreadAllocator select_on_container_copy_construction() const noexcept {
    return *this;
  }

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;
  using is_always_equal = std::is_empty<ThreadAllocator>;
};

template <class T, class U>
constexpr bool operator==(
    ThreadAllocator<T> const&, ThreadAllocator<U> const&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(
    ThreadAllocator<T> const& x, ThreadAllocator<U> const& y) noexcept {
  return !(x == y);
}
}  // namespace fmpi
#endif
