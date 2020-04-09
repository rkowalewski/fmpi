#ifndef FMPI_ALLOCATOR_BUFFERALLOCATOR_HPP
#define FMPI_ALLOCATOR_BUFFERALLOCATOR_HPP

#include <cstddef>
#include <vector>

#include <fmpi/allocator/ContiguousPoolAllocator.hpp>
namespace fmpi {

template <
    class T,
    bool ThreadSafe      = false,
    class SmallAllocator = ContiguousPoolAllocator<T, ThreadSafe>>
class BufferAllocator {
  std::size_t const small_threshold_{};
  SmallAllocator    small_alloc_{};

 public:
  using value_type = T;

  //     using pointer       = value_type*;
  //     using const_pointer = typename std::pointer_traits<pointer>::template
  //                                                     rebind<value_type
  //                                                     const>;
  //     using void_pointer       = typename
  //     std::pointer_traits<pointer>::template
  //                                                           rebind<void>;
  //     using const_void_pointer = typename
  //     std::pointer_traits<pointer>::template
  //                                                           rebind<const
  //                                                           void>;

  //     using difference_type = typename
  //     std::pointer_traits<pointer>::difference_type; using size_type =
  //     std::make_unsigned_t<difference_type>;

  //     template <class U> struct rebind {typedef BufferAllocator<U> other;};

  BufferAllocator(std::size_t small_threshold, SmallAllocator alloc) noexcept
    : small_threshold_{small_threshold}
    , small_alloc_{alloc} {
  }  // not required, unless used

#if 0
  template <class U>
  BufferAllocator(BufferAllocator<U> const&) noexcept {
  }
#endif

  value_type*  // Use pointer if pointer is not a value_type*
  allocate(std::size_t n) {
    if (n < small_threshold_) {
      return small_alloc_.allocate(n);
    } else {
      return static_cast<value_type*>(::operator new(n * sizeof(value_type)));
    }
  }

  void deallocate(
      value_type* p,
      std::size_t) noexcept  // Use pointer if pointer is not a value_type*
  {
    if (small_alloc_.isManaged(p)) {
      small_alloc_.deallocate(p);
    } else {
      ::operator delete(p);
    }
  }

  //     value_type*
  //     allocate(std::size_t n, const_void_pointer)
  //     {
  //         return allocate(n);
  //     }

  //     template <class U, class ...Args>
  //     void
  //     construct(U* p, Args&& ...args)
  //     {
  //         ::new(p) U(std::forward<Args>(args)...);
  //     }

  //     template <class U>
  //     void
  //     destroy(U* p) noexcept
  //     {
  //         p->~U();
  //     }

  //     std::size_t
  //     max_size() const noexcept
  //     {
  //         return std::numeric_limits<size_type>::max();
  //     }

  //     BufferAllocator
  //     select_on_container_copy_construction() const
  //     {
  //         return *this;
  //     }

  //     using propagate_on_container_copy_assignment = std::false_type;
  //     using propagate_on_container_move_assignment = std::false_type;
  //     using propagate_on_container_swap            = std::false_type;
  //     using is_always_equal                        =
  //     std::is_empty<BufferAllocator>;
};

template <class T, class U>
bool operator==(
    BufferAllocator<T> const&, BufferAllocator<U> const&) noexcept {
  return false;
}

template <class T, class U>
bool operator!=(
    BufferAllocator<T> const& x, BufferAllocator<U> const& y) noexcept {
  return !(x == y);
}
}  // namespace fmpi

#endif
