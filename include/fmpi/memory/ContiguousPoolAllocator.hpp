#ifndef FMPI_MEMORY_CONTIGUOUSPOOLALLOCATOR_HPP
#define FMPI_MEMORY_CONTIGUOUSPOOLALLOCATOR_HPP

#include <memory>

namespace fmpi {

//==============================================================================
//                        struct ContiguousPoolAllocator
//==============================================================================
/// @struct ContiguousPoolAllocator.
/// @brief Provides fast (quasi zero-time) in-place allocation for STL
/// containers.
///        Objects are allocated from a contiguous buffer (aka object pool).
///        When the buffer is exhausted, allocation is delegated to the heap.
/// @tparam T The type to allocate.
/// @note This allocator is not thread safe.
template <typename T>
struct ContiguousPoolAllocator {
  template <typename U>
  friend struct ContiguousPoolAllocator;
  //------------------------------ Typedefs ----------------------------------
  typedef ContiguousPoolAllocator<T> this_type;
  typedef T                          value_type;
  typedef value_type*                pointer;
  typedef const value_type*          const_pointer;
  typedef value_type&                reference;
  typedef const value_type&          const_reference;
  typedef size_t                     size_type;
  typedef uint16_t                   index_type;
  typedef std::ptrdiff_t             difference_type;
  typedef std::true_type             propagate_on_container_move_assignment;
  typedef std::true_type             propagate_on_container_copy_assignment;
  typedef std::true_type             propagate_on_container_swap;
  typedef std::false_type            is_always_equal;
  typedef std::false_type            default_constructor;
  typedef std::aligned_storage<sizeof(T), alignof(T)> storage_type;
  typedef typename storage_type::type                 aligned_type;

  //------------------------------- Methods ----------------------------------
  ContiguousPoolAllocator();
  ContiguousPoolAllocator(aligned_type* buffer, index_type size);

  template <typename U>
  struct rebind {
    typedef ContiguousPoolAllocator<U> other;
  };
  // Rebound types
  template <typename U>
  explicit ContiguousPoolAllocator(const ContiguousPoolAllocator<U>& other);
  template <typename U>
  explicit ContiguousPoolAllocator(ContiguousPoolAllocator<U>&& other);
  template <typename U>
  ContiguousPoolAllocator& operator=(const ContiguousPoolAllocator<U>&) =
      delete;
  template <typename U>
  ContiguousPoolAllocator& operator=(ContiguousPoolAllocator<U>&&) = delete;

  static ContiguousPoolAllocator select_on_container_copy_construction(
      const ContiguousPoolAllocator& other) {
    return ContiguousPoolAllocator(other);
  }
  bool operator==(const this_type& other) const {
    return _control && other._control &&
           (_control->_buffer == other._control->_buffer);
  }
  bool operator!=(const this_type& other) const {
    return !operator==(other);
  }

  //------------------------- Accessors ------------------------------
  void    setBuffer(aligned_type* buffer, index_type size);
  pointer address(reference x) const;
  [[nodiscard]] const_pointer address(const_reference x) const;
  [[nodiscard]] size_type     max_size() const;
  template <typename... Args>
  void    construct(T* p, Args&&... args);
  void    destroy(pointer p);
  pointer allocate(size_type /*n*/ = 1, const_pointer /*unused*/ = nullptr);
  void    deallocate(pointer p, size_type /*n*/ = 1);
  template <typename... Args>
  pointer                  create(Args&&... args);
  void                     dispose(pointer p);
  [[nodiscard]] size_t     allocatedBlocks() const;
  [[nodiscard]] size_t     allocatedHeapBlocks() const;
  [[nodiscard]] bool       isFull() const;
  [[nodiscard]] bool       isEmpty() const;
  [[nodiscard]] index_type size() const;
  explicit                 operator bool() const;

  bool isManaged(pointer p);

 private:
  pointer    bufferStart();
  pointer    bufferEnd();
  index_type blockIndex(pointer p);
  bool       findContiguous(index_type n);

  //------------------------------- Members ----------------------------------
  struct Control {
    ~Control() {
      delete[] _freeBlocks;
    }
    index_type    _size{0};
    aligned_type* _buffer{nullptr};  // non-owning
    index_type*   _freeBlocks{nullptr};
    ssize_t       _freeBlockIndex{-1};
    size_t        _numHeapAllocatedBlocks{0};
  };
  std::shared_ptr<Control> _control;
};

template <typename U, typename T>
size_t resize(size_t t_size) {
  return (t_size * sizeof(U)) / sizeof(T);
}

template <typename T>
ContiguousPoolAllocator<T>::ContiguousPoolAllocator()
  : _control(std::make_shared<Control>()) {
}

template <typename T>
ContiguousPoolAllocator<T>::ContiguousPoolAllocator(
    aligned_type* buffer, index_type size)
  : _control(std::make_shared<Control>()) {
  setBuffer(buffer, size);
}

template <typename T>
template <typename U>
ContiguousPoolAllocator<T>::ContiguousPoolAllocator(
    const ContiguousPoolAllocator<U>& other)
  : _control(std::reinterpret_pointer_cast<Control>(other._control)) {
  if (!_control || !_control->_buffer) {
    throw std::runtime_error("Invalid allocator.");
  }
  // normalize size of buffer
  index_type newSize =
      std::min(_control->_size, (index_type)resize<U, T>(_control->_size));
  _control->_size           = newSize;  // resize buffer
  _control->_freeBlockIndex = newSize - 1;
}

template <typename T>
template <typename U>
ContiguousPoolAllocator<T>::ContiguousPoolAllocator(
    ContiguousPoolAllocator<U>&& other)
  : _control(std::move(other._control)) {
  if (!_control || !_control->_buffer) {
    throw std::runtime_error("Invalid allocator.");
  }
  // normalize size of buffer
  index_type newSize =
      std::min(_control->_size, (index_type)resize<U, T>(_control->_size));
  _control->_size           = newSize;  // resize buffer
  _control->_freeBlockIndex = newSize - 1;
}

template <typename T>
void ContiguousPoolAllocator<T>::setBuffer(
    aligned_type* buffer, index_type size) {
  if (!_control) {
    throw std::bad_alloc();
  }
  if (!buffer) {
    throw std::runtime_error("Null buffer");
  }
  if (size == 0) {
    throw std::runtime_error("Invalid allocator pool size");
  }
  _control->_size   = size;
  _control->_buffer = buffer;

  delete[] _control->_freeBlocks;

  _control->_freeBlocks = new index_type[size];
  if (!_control->_freeBlocks) {
    throw std::bad_alloc();
  }
  // build the free stack
  for (index_type i = 0; i < size; ++i) {
    _control->_freeBlocks[i] = i;
  }
  _control->_freeBlockIndex = size - 1;
}

template <typename T>
typename ContiguousPoolAllocator<T>::pointer
ContiguousPoolAllocator<T>::address(reference x) const {
  return &x;
}

template <typename T>
typename ContiguousPoolAllocator<T>::const_pointer
ContiguousPoolAllocator<T>::address(const_reference x) const {
  return &x;
}

template <typename T>
typename ContiguousPoolAllocator<T>::size_type
ContiguousPoolAllocator<T>::max_size() const {
  return 1;  // only 1 supported for now
}

template <typename T>
template <typename... Args>
void ContiguousPoolAllocator<T>::construct(T* p, Args&&... args) {
  new ((void*)p) T(std::forward<Args>(args)...);  // construct in-place
}

template <typename T>
void ContiguousPoolAllocator<T>::destroy(pointer p) {
  if (p != nullptr) {
    p->~T();
  }
}

template <typename T>
typename ContiguousPoolAllocator<T>::pointer
ContiguousPoolAllocator<T>::allocate(size_type n, const_pointer /*unused*/) {
  assert(bufferStart());
  {
    constexpr auto max_n =
        std::size_t{std::numeric_limits<index_type>::max()};
    if (n < max_n && findContiguous(static_cast<index_type>(n))) {
      _control->_freeBlockIndex -= (n - 1);
      return reinterpret_cast<pointer>(
          &_control
               ->_buffer[_control->_freeBlocks[_control->_freeBlockIndex--]]);
    }
    // Use heap allocation
    ++_control->_numHeapAllocatedBlocks;
  }
  return static_cast<pointer>(::operator new(n * sizeof(value_type)));
}

template <typename T>
void ContiguousPoolAllocator<T>::deallocate(pointer p, size_type n) {
  if (p == nullptr) {
    return;
  }
  assert(bufferStart());
  if (isManaged(p)) {
    // find index of the block and return the individual blocks to the free
    // pool
    for (size_type i = 0; i < n; ++i) {
      _control->_freeBlocks[++_control->_freeBlockIndex] = blockIndex(p + i);
    }
  } else {
    ::operator delete(p);

    --_control->_numHeapAllocatedBlocks;
  }
}

template <typename T>
template <typename... Args>
typename ContiguousPoolAllocator<T>::pointer
ContiguousPoolAllocator<T>::create(Args&&... args) {
  T* p = allocate();
  construct(p, std::forward<Args>(args)...);
  return p;
}

template <typename T>
void ContiguousPoolAllocator<T>::dispose(pointer p) {
  destroy(p);
  deallocate(p);
}

template <typename T>
size_t ContiguousPoolAllocator<T>::allocatedBlocks() const {
  return _control->_size ? _control->_size - _control->_freeBlockIndex - 1
                         : 0;
}

template <typename T>
size_t ContiguousPoolAllocator<T>::allocatedHeapBlocks() const {
  return _control->_numHeapAllocatedBlocks;
}

template <typename T>
bool ContiguousPoolAllocator<T>::isFull() const {
  return _control->_freeBlockIndex == _control->_size - 1;
}

template <typename T>
bool ContiguousPoolAllocator<T>::isEmpty() const {
  return _control->_freeBlockIndex == -1;
}

template <typename T>
typename ContiguousPoolAllocator<T>::index_type
ContiguousPoolAllocator<T>::size() const {
  return _control->_size;
}

template <typename T>
ContiguousPoolAllocator<T>::operator bool() const {
  return _control != nullptr;
}

template <typename T>
typename ContiguousPoolAllocator<T>::pointer
ContiguousPoolAllocator<T>::bufferStart() {
  return reinterpret_cast<pointer>(_control->_buffer);
}

template <typename T>
typename ContiguousPoolAllocator<T>::pointer
ContiguousPoolAllocator<T>::bufferEnd() {
  return reinterpret_cast<pointer>(_control->_buffer + _control->_size);
}

template <typename T>
bool ContiguousPoolAllocator<T>::isManaged(pointer p) {
  return (bufferStart() <= p) && (p < bufferEnd());
}

template <typename T>
typename ContiguousPoolAllocator<T>::index_type
ContiguousPoolAllocator<T>::blockIndex(pointer p) {
  return static_cast<index_type>(
      reinterpret_cast<aligned_type*>(p) - _control->_buffer);
}

template <typename T>
bool ContiguousPoolAllocator<T>::findContiguous(index_type n) {
  if ((_control->_freeBlockIndex + 1) < n) {
    return false;
  }
  bool          found = true;
  aligned_type* last =
      &_control->_buffer[_control->_freeBlocks[_control->_freeBlockIndex]];
  for (ssize_t i = _control->_freeBlockIndex - 1;
       i > _control->_freeBlockIndex - n;
       --i) {
    aligned_type* first = &_control->_buffer[_control->_freeBlocks[i]];
    if ((last - first) != (_control->_freeBlockIndex - i)) {
      return false;
    }
  }
  return found;
}

}  // namespace fmpi
#endif
