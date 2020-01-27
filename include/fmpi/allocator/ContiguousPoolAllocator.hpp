#ifndef FMPI_ALLOCATOR_CONTIGUOUSPOOLMANAGER_HPP
#define FMPI_ALLOCATOR_CONTIGUOUSPOOLMANAGER_HPP

#include <memory>
#include <mutex>

namespace fmpi {

namespace detail {

template <bool ThreadSafe>
struct LockGuard : public std::lock_guard<std::mutex> {
  using std::lock_guard<std::mutex>::lock_guard;
};

template <>
struct LockGuard<false> {
  LockGuard(std::mutex& /*unused*/) {
  }
};
}  // namespace detail

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
template <typename T, bool ThreadSafe = false>
struct ContiguousPoolAllocator {
  template <typename U, bool V>
  friend struct ContiguousPoolAllocator;
  //------------------------------ Typedefs ----------------------------------
  typedef ContiguousPoolAllocator<T, ThreadSafe> this_type;
  typedef T                                      value_type;
  typedef value_type*                            pointer;
  typedef const value_type*                      const_pointer;
  typedef value_type&                            reference;
  typedef const value_type&                      const_reference;
  typedef size_t                                 size_type;
  typedef uint16_t                               index_type;
  typedef std::ptrdiff_t                         difference_type;
  typedef std::true_type  propagate_on_container_move_assignment;
  typedef std::true_type  propagate_on_container_copy_assignment;
  typedef std::true_type  propagate_on_container_swap;
  typedef std::false_type is_always_equal;
  typedef std::false_type default_constructor;
  typedef std::aligned_storage<sizeof(T), alignof(T)> storage_type;
  typedef typename storage_type::type                 aligned_type;

  //------------------------------- Methods ----------------------------------
  ContiguousPoolAllocator();
  ContiguousPoolAllocator(aligned_type* buffer, index_type size);

  template <typename U>
  struct rebind {
    typedef ContiguousPoolAllocator<U, ThreadSafe> other;
  };
  // Rebound types
  template <typename U>
  ContiguousPoolAllocator(
      const ContiguousPoolAllocator<U, ThreadSafe>& other);
  template <typename U>
  ContiguousPoolAllocator(ContiguousPoolAllocator<U, ThreadSafe>&& other);
  template <typename U>
  ContiguousPoolAllocator& operator                  =(
      const ContiguousPoolAllocator<U, ThreadSafe>&) = delete;
  template <typename U>
  ContiguousPoolAllocator& operator             =(
      ContiguousPoolAllocator<U, ThreadSafe>&&) = delete;

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
  void          setBuffer(aligned_type* buffer, index_type size);
  pointer       address(reference x) const;
  const_pointer address(const_reference x) const;
  size_type     max_size() const;
  template <typename... Args>
  void    construct(T* p, Args&&... args);
  void    destroy(pointer p);
  pointer allocate(size_type = 1, const_pointer = 0);
  void    deallocate(pointer p, size_type = 1);
  template <typename... Args>
  pointer    create(Args&&... args);
  void       dispose(pointer p);
  size_t     allocatedBlocks() const;
  size_t     allocatedHeapBlocks() const;
  bool       isFull() const;
  bool       isEmpty() const;
  index_type size() const;
  explicit   operator bool() const;

 private:
  pointer    bufferStart();
  pointer    bufferEnd();
  bool       isManaged(pointer p);
  index_type blockIndex(pointer p);
  bool       findContiguous(index_type n);

  //------------------------------- Members ----------------------------------
  struct Control {
    ~Control() {
      delete[] _freeBlocks;
    }
    index_type         _size{0};
    aligned_type*      _buffer{nullptr};  // non-owning
    index_type*        _freeBlocks{nullptr};
    ssize_t            _freeBlockIndex{-1};
    size_t             _numHeapAllocatedBlocks{0};
    mutable std::mutex _mutex;
  };
  std::shared_ptr<Control> _control;
};

template <typename U, typename T>
size_t resize(size_t t_size) {
  return (t_size * sizeof(U)) / sizeof(T);
}

template <typename T, bool ThreadSafe>
ContiguousPoolAllocator<T, ThreadSafe>::ContiguousPoolAllocator()
  : _control(std::make_shared<Control>()) {
}

template <typename T, bool ThreadSafe>
ContiguousPoolAllocator<T, ThreadSafe>::ContiguousPoolAllocator(
    aligned_type* buffer, index_type size)
  : _control(std::make_shared<Control>()) {
  setBuffer(buffer, size);
}

template <typename T, bool ThreadSafe>
template <typename U>
ContiguousPoolAllocator<T, ThreadSafe>::ContiguousPoolAllocator(
    const ContiguousPoolAllocator<U, ThreadSafe>& other)
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

template <typename T, bool ThreadSafe>
template <typename U>
ContiguousPoolAllocator<T, ThreadSafe>::ContiguousPoolAllocator(
    ContiguousPoolAllocator<U, ThreadSafe>&& other)
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

template <typename T, bool ThreadSafe>
void ContiguousPoolAllocator<T, ThreadSafe>::setBuffer(
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
  if (_control->_freeBlocks) {
    delete[] _control->_freeBlocks;
  }
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

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::pointer
ContiguousPoolAllocator<T, ThreadSafe>::address(reference x) const {
  return &x;
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::const_pointer
ContiguousPoolAllocator<T, ThreadSafe>::address(const_reference x) const {
  return &x;
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::size_type
ContiguousPoolAllocator<T, ThreadSafe>::max_size() const {
  return 1;  // only 1 supported for now
}

template <typename T, bool ThreadSafe>
template <typename... Args>
void ContiguousPoolAllocator<T, ThreadSafe>::construct(T* p, Args&&... args) {
  new ((void*)p) T(std::forward<Args>(args)...);  // construct in-place
}

template <typename T, bool ThreadSafe>
void ContiguousPoolAllocator<T, ThreadSafe>::destroy(pointer p) {
  if (p != nullptr) {
    p->~T();
  }
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::pointer
ContiguousPoolAllocator<T, ThreadSafe>::allocate(size_type n, const_pointer) {
  assert(bufferStart());
  {
    detail::LockGuard<ThreadSafe> lock(_control->_mutex);
    if (findContiguous(static_cast<index_type>(n))) {
      _control->_freeBlockIndex -= (n - 1);
      return reinterpret_cast<pointer>(
          &_control
               ->_buffer[_control->_freeBlocks[_control->_freeBlockIndex--]]);
    }
    // Use heap allocation
    ++_control->_numHeapAllocatedBlocks;
  }
  return (pointer) new char[sizeof(value_type) * n];
}

template <typename T, bool ThreadSafe>
void ContiguousPoolAllocator<T, ThreadSafe>::deallocate(
    pointer p, size_type n) {
  if (p == nullptr) {
    return;
  }
  assert(bufferStart());
  if (isManaged(p)) {
    // find index of the block and return the individual blocks to the free
    // pool
    detail::LockGuard<ThreadSafe> lock(_control->_mutex);
    for (size_type i = 0; i < n; ++i) {
      _control->_freeBlocks[++_control->_freeBlockIndex] = blockIndex(p + i);
    }
  } else {
    delete[](char*) p;
    detail::LockGuard<ThreadSafe> lock(_control->_mutex);
    --_control->_numHeapAllocatedBlocks;
  }
}

template <typename T, bool ThreadSafe>
template <typename... Args>
typename ContiguousPoolAllocator<T, ThreadSafe>::pointer
ContiguousPoolAllocator<T, ThreadSafe>::create(Args&&... args) {
  T* p = allocate();
  construct(p, std::forward<Args>(args)...);
  return p;
}

template <typename T, bool ThreadSafe>
void ContiguousPoolAllocator<T, ThreadSafe>::dispose(pointer p) {
  destroy(p);
  deallocate(p);
}

template <typename T, bool ThreadSafe>
size_t ContiguousPoolAllocator<T, ThreadSafe>::allocatedBlocks() const {
  return _control->_size ? _control->_size - _control->_freeBlockIndex - 1
                         : 0;
}

template <typename T, bool ThreadSafe>
size_t ContiguousPoolAllocator<T, ThreadSafe>::allocatedHeapBlocks() const {
  return _control->_numHeapAllocatedBlocks;
}

template <typename T, bool ThreadSafe>
bool ContiguousPoolAllocator<T, ThreadSafe>::isFull() const {
  return _control->_freeBlockIndex == _control->_size - 1;
}

template <typename T, bool ThreadSafe>
bool ContiguousPoolAllocator<T, ThreadSafe>::isEmpty() const {
  return _control->_freeBlockIndex == -1;
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::index_type
ContiguousPoolAllocator<T, ThreadSafe>::size() const {
  return _control->_size;
}

template <typename T, bool ThreadSafe>
ContiguousPoolAllocator<T, ThreadSafe>::operator bool() const {
  return _control != nullptr;
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::pointer
ContiguousPoolAllocator<T, ThreadSafe>::bufferStart() {
  return reinterpret_cast<pointer>(_control->_buffer);
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::pointer
ContiguousPoolAllocator<T, ThreadSafe>::bufferEnd() {
  return reinterpret_cast<pointer>(_control->_buffer + _control->_size);
}

template <typename T, bool ThreadSafe>
bool ContiguousPoolAllocator<T, ThreadSafe>::isManaged(pointer p) {
  return (bufferStart() <= p) && (p < bufferEnd());
}

template <typename T, bool ThreadSafe>
typename ContiguousPoolAllocator<T, ThreadSafe>::index_type
ContiguousPoolAllocator<T, ThreadSafe>::blockIndex(pointer p) {
  return static_cast<index_type>(
      reinterpret_cast<aligned_type*>(p) - _control->_buffer);
}

template <typename T, bool ThreadSafe>
bool ContiguousPoolAllocator<T, ThreadSafe>::findContiguous(index_type n) {
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
