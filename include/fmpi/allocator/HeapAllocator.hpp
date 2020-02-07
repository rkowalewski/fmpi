#ifndef FMPI_ALLOCATOR_HEAPALLOCATOR_HPP
#define FMPI_ALLOCATOR_HEAPALLOCATOR_HPP

#include <fmpi/allocator/ContiguousPoolAllocator.hpp>

namespace fmpi {

//==============================================================================
//                            struct HeapAllocator
//==============================================================================
/// @struct HeapAllocator.
/// @brief Provides a heap-based object pool to the underlying
/// ContiguousPoolManager.
/// @tparam T The type to allocate.
/// @note This allocator is not thread safe.
template <typename T, bool ThreadSafe = false>
struct HeapAllocator : public ContiguousPoolAllocator<T, ThreadSafe> {
  //------------------------------ Typedefs ----------------------------------
  typedef HeapAllocator<T, ThreadSafe> this_type;
  typedef T                            value_type;
  typedef value_type*                  pointer;
  typedef const value_type*            const_pointer;
  typedef value_type&                  reference;
  typedef const value_type&            const_reference;
  typedef size_t                       size_type;
  typedef uint16_t                     index_type;
  typedef std::ptrdiff_t               difference_type;
  typedef std::true_type               propagate_on_container_move_assignment;
  typedef std::false_type              propagate_on_container_copy_assignment;
  typedef std::true_type               propagate_on_container_swap;
  typedef std::false_type              is_always_equal;
  typedef std::false_type              default_constructor;
  typedef std::aligned_storage<sizeof(value_type), alignof(value_type)>
                                      storage_type;
  typedef typename storage_type::type aligned_type;

  template <typename U>
  struct rebind {
    typedef HeapAllocator<U> other;
  };
  //------------------------------- Methods ----------------------------------
  explicit HeapAllocator(index_type size)
    : _size(size)
    , _buffer(std::make_unique<aligned_type[]>(size)) {
    if (!_buffer) {
      throw std::bad_alloc();
    }
    this->setBuffer(_buffer.get(), _size);
  }
  HeapAllocator(const this_type& other)
    : HeapAllocator(other._size) {
  }
  HeapAllocator(this_type&& other) = default;
  HeapAllocator& operator=(const this_type&) = delete;
  HeapAllocator& operator=(this_type&& other) = delete;

  // Rebound types
  template <typename U>
  explicit HeapAllocator(const HeapAllocator<U>& other)
    : HeapAllocator(other.size()) {
  }
  template <typename U>
  explicit HeapAllocator(HeapAllocator<U>&& other)
    : ContiguousPoolAllocator<T>(std::move(other))
    , _size((index_type)resize<U, T>(other._size))
    , _buffer(other._buffer) {
    other._size   = 0;
    other._buffer = nullptr;
  }
  template <typename U>
  HeapAllocator& operator=(const HeapAllocator<U>&) = delete;
  template <typename U>
  HeapAllocator& operator=(HeapAllocator<U>&&) = delete;

  static HeapAllocator select_on_container_copy_construction(
      const HeapAllocator& other) {
    return HeapAllocator(other.size());
  }
  bool operator==(const this_type& /*unused*/) const {
    return true;
  }
  bool operator!=(const this_type& /*unused*/) const {
    return false;
  }
  [[nodiscard]] index_type size() const {
    return _size;
  }

 private:
  //------------------------------- Members ----------------------------------
  index_type                      _size{};
  std::unique_ptr<aligned_type[]> _buffer;
};
}  // namespace fmpi
#endif
