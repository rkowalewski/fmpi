#ifndef FMPI_MEMORY_MPIALLOCATOR_HPP
#define FMPI_MEMORY_MPIALLOCATOR_HPP
#include <mpi.h>

#include <cstddef>
#include <memory>
namespace fmpi {
template <class T>
class MpiAllocator {
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
  using is_always_equal                        = std::is_empty<MpiAllocator>;

  template <typename U>
  struct rebind {
    typedef MpiAllocator<U> other;
  };

  MpiAllocator() noexcept = default;

  template <typename U>
  explicit MpiAllocator(const MpiAllocator<U>& /*unused*/) noexcept {
  }

  template <typename U>
  explicit MpiAllocator(const MpiAllocator<U>&& /*unused*/) noexcept {
  }

  pointer allocate(std::size_t n) {
    pointer result;
    MPI_Alloc_mem(
        static_cast<MPI_Aint>(n * sizeof(T)), MPI_INFO_NULL, &result);
    return result;
  }

  void deallocate(value_type* p, std::size_t /*size*/) {
    MPI_Free_mem(p);
  }
};

template <class T, class U>
constexpr bool operator==(
    MpiAllocator<T> const& /*unused*/,
    MpiAllocator<U> const& /*unused*/) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(
    MpiAllocator<T> const& x, MpiAllocator<U> const& y) noexcept {
  return !(x == y);
}
}  // namespace fmpi
#endif
