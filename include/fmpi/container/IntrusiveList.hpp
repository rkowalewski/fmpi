#ifndef FMPI_CONTAINER_INTRUSIVELIST_HPP
#define FMPI_CONTAINER_INTRUSIVELIST_HPP

#include <atomic>
#include <cassert>
#include <exception>
#include <forward_list>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>

namespace fmpi {

template <class T>
struct intrusive_ptr {
  using value_type   = T;
  using element_type = T;
  using pointer      = T*;
  using reference    = T&;

  template <class U>
  friend struct intrusive_ptr;

  intrusive_ptr() noexcept
    : ptr_{nullptr} {
  }

  explicit intrusive_ptr(std::nullptr_t) noexcept
    : ptr_{nullptr} {
  }

  explicit intrusive_ptr(pointer p) noexcept
    : ptr_{p} {
    if (ptr_) {
      ptr_->addref();
    }
  }

  intrusive_ptr(intrusive_ptr const& p) noexcept
    : ptr_{p.ptr_} {
    if (ptr_) {
      ptr_->addref();
    }
  }

  intrusive_ptr(intrusive_ptr&& p) noexcept
    : ptr_{p.ptr_} {
    p.ptr_ = nullptr;
  }

  template <class U>
  explicit intrusive_ptr(intrusive_ptr<U> const& p) noexcept
    : ptr_{p.ptr_} {
    if (ptr_) {
      ptr_->addref();
    }
  }

  template <class U>
  explicit intrusive_ptr(intrusive_ptr<U>&& p) noexcept
    : ptr_{p.ptr_} {
    p.ptr_ = nullptr;
  }

  ~intrusive_ptr() noexcept {
    if (ptr_) {
      ptr_->release();
    }
  }

  pointer get() const noexcept {
    return ptr_;
  }

  pointer detach() noexcept {
    pointer p = ptr_;
    ptr_      = nullptr;
    return p;
  }

  explicit operator pointer() const noexcept {
    return ptr_;
  }

  pointer operator->() const noexcept {
    assert(ptr_ != nullptr);
    return ptr_;
  }

  reference operator*() const noexcept {
    assert(ptr_ != nullptr);
    return *ptr_;
  }

  pointer* operator&() noexcept {
    assert(ptr_ == nullptr);
    return &ptr_;
  }

  pointer const* operator&() const noexcept {
    return &ptr_;
  }

  intrusive_ptr& operator=(pointer p) noexcept {
    if (p) {
      p->addref();
    }
    pointer o = ptr_;
    ptr_      = p;
    if (o) {
      o->release();
    }
    return *this;
  }

  intrusive_ptr& operator=(std::nullptr_t) noexcept {
    if (ptr_) {
      ptr_->release();
      ptr_ = nullptr;
    }
    return *this;
  }

  intrusive_ptr& operator=(intrusive_ptr const& p) noexcept {
    return (*this = p.ptr_);
  }

  intrusive_ptr& operator=(intrusive_ptr&& p) noexcept {
    if (ptr_) {
      ptr_->release();
    }
    ptr_   = p.ptr_;
    p.ptr_ = nullptr;
    return *this;
  }

  template <class U>
  intrusive_ptr& operator=(intrusive_ptr<U> const& p) noexcept {
    return (*this = p.ptr_);
  }

  template <class U>
  intrusive_ptr& operator=(intrusive_ptr<U>&& p) noexcept {
    if (ptr_) {
      ptr_->release();
    }
    ptr_   = p.ptr_;
    p.ptr_ = nullptr;
    return *this;
  }

  void swap(pointer* pp) noexcept {
    pointer p = ptr_;
    ptr_      = *pp;
    *pp       = p;
  }

  void swap(intrusive_ptr& p) noexcept {
    swap(&p.ptr_);
  }

 private:
  pointer ptr_;
};

template <class T, class U>
bool operator==(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() == b.get();
}

template <class T, class U>
bool operator==(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() == b;
}

template <class T, class U>
bool operator==(T* a, intrusive_ptr<U> const& b) noexcept {
  return a == b.get();
}

template <class T, class U>
bool operator!=(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() != b.get();
}

template <class T, class U>
bool operator!=(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() != b;
}

template <class T, class U>
bool operator!=(T* a, intrusive_ptr<U> const& b) noexcept {
  return a != b.get();
}

template <class T, class U>
bool operator<(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() < b.get();
}

template <class T, class U>
bool operator<(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() < b;
}

template <class T, class U>
bool operator<(T* a, intrusive_ptr<U> const& b) noexcept {
  return a < b.get();
}

template <class T, class U>
bool operator<=(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() <= b.get();
}

template <class T, class U>
bool operator<=(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() <= b;
}

template <class T, class U>
bool operator<=(T* a, intrusive_ptr<U> const& b) noexcept {
  return a <= b.get();
}

template <class T, class U>
bool operator>(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() > b.get();
}

template <class T, class U>
bool operator>(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() > b;
}

template <class T, class U>
bool operator>(T* a, intrusive_ptr<U> const& b) noexcept {
  return a > b.get();
}

template <class T, class U>
bool operator>=(
    intrusive_ptr<T> const& a, intrusive_ptr<U> const& b) noexcept {
  return a.get() >= b.get();
}

template <class T, class U>
bool operator>=(intrusive_ptr<T> const& a, U* b) noexcept {
  return a.get() >= b;
}

template <class T, class U>
bool operator>=(T* a, intrusive_ptr<U> const& b) noexcept {
  return a >= b.get();
}

template <class T>
bool operator==(intrusive_ptr<T> const& a, std::nullptr_t) noexcept {
  return a.get() == nullptr;
}

template <class T>
bool operator==(std::nullptr_t, intrusive_ptr<T> const& b) noexcept {
  return nullptr == b.get();
}

template <class T>
bool operator!=(intrusive_ptr<T> const& a, std::nullptr_t) noexcept {
  return a.get() != nullptr;
}

template <class T>
bool operator!=(std::nullptr_t, intrusive_ptr<T> const& b) noexcept {
  return nullptr != b.get();
}

template <class T>
T* get_pointer(intrusive_ptr<T> const& p) noexcept {
  return p.get();
}

template <class T, class U>
intrusive_ptr<U> static_pointer_cast(intrusive_ptr<T> const& p) noexcept {
  return intrusive_ptr<U>{static_cast<U*>(p.get())};
}

template <class T, class U>
intrusive_ptr<U> const_pointer_cast(intrusive_ptr<T> const& p) noexcept {
  return intrusive_ptr<U>{const_cast<U*>(p.get())};
}

template <class T, class U>
intrusive_ptr<U> dynamic_pointer_cast(intrusive_ptr<T> const& p) noexcept {
  return intrusive_ptr<U>{dynamic_cast<U*>(p.get())};
}

struct ref_count {
  unsigned long addref() noexcept {
    return ++count_;
  }

  unsigned long release() noexcept {
    return --count_;
  }

  [[nodiscard]] unsigned long get() const noexcept {
    return count_;
  }

 private:
  unsigned long count_{0};
};

struct ref_count_atomic {
  unsigned long addref() noexcept {
    return ++count_;
  }

  unsigned long release() noexcept {
    return --count_;
  }

  [[nodiscard]] unsigned long get() const noexcept {
    return count_.load();
  }

 private:
  std::atomic<unsigned long> count_{0};
};

template <class Class, class RefCount = ref_count>
struct ref_counted {
  ref_counted() noexcept = default;

  ref_counted(ref_counted const& /*unused*/) noexcept {
  }

  ref_counted& operator=(ref_counted const& /*unused*/) noexcept {
    return *this;
  }

  void addref() noexcept {
    count_.addref();
  }

  void release() noexcept {
    if (count_.release() == 0) {
      delete static_cast<Class*>(this);
    }
  }

 protected:
  ~ref_counted() noexcept = default;

 private:
  RefCount count_{};
};

template <class T>
class stable_list {
  struct link_element : ref_counted<link_element> {
    link_element() noexcept = default;

    ~link_element() noexcept {
      if (next) {       // If we have a next element upon destruction
        value()->~T();  // then this link is used, else it's a dummy
      }
    }

    template <class... Args>
    void construct(Args&&... args) {
      new (storage()) T{std::forward<Args>(args)...};
    }

    T* value() noexcept {
      return static_cast<T*>(storage());
    }

    void* storage() noexcept {
      return static_cast<void*>(&buffer);
    }

    intrusive_ptr<link_element> next;
    intrusive_ptr<link_element> prev;

    std::aligned_storage_t<sizeof(T), std::alignment_of<T>::value> buffer;
  };

  intrusive_ptr<link_element> head_;
  intrusive_ptr<link_element> tail_;

  std::size_t elements_{};

 public:
  template <class U>
  struct iterator_base {
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type        = std::remove_const_t<U>;
    using difference_type   = ptrdiff_t;
    using reference         = U&;
    using pointer           = U*;

    template <class V>
    friend class stable_list;

    iterator_base() noexcept  = default;
    ~iterator_base() noexcept = default;

    iterator_base(iterator_base const& i) noexcept
      : element_{i.element_} {
    }

    iterator_base(iterator_base&& i) noexcept
      : element_{std::move(i.element_)} {
    }

    template <class V>
    explicit iterator_base(iterator_base<V> const& i) noexcept
      : element_{i.element_} {
    }

    template <class V>
    explicit iterator_base(iterator_base<V>&& i) noexcept
      : element_{std::move(i.element_)} {
    }

    iterator_base& operator=(iterator_base const& i) noexcept {
      element_ = i.element_;
      return *this;
    }

    iterator_base& operator=(iterator_base&& i) noexcept {
      element_ = std::move(i.element_);
      return *this;
    }

    template <class V>
    iterator_base& operator=(iterator_base<V> const& i) noexcept {
      element_ = i.element_;
      return *this;
    }

    template <class V>
    iterator_base& operator=(iterator_base<V>&& i) noexcept {
      element_ = std::move(i.element_);
      return *this;
    }

    iterator_base& operator++() noexcept {
      element_ = element_->next;
      return *this;
    }

    const iterator_base operator++(int) noexcept {
      iterator_base i{*this};
      ++(*this);
      return i;
    }

    iterator_base& operator--() noexcept {
      element_ = element_->prev;
      return *this;
    }

    const iterator_base operator--(int) noexcept {
      iterator_base i{*this};
      --(*this);
      return i;
    }

    reference operator*() const noexcept {
      return *element_->value();
    }

    pointer operator->() const noexcept {
      return element_->value();
    }

    template <class V>
    bool operator==(iterator_base<V> const& i) const noexcept {
      return element_ == i.element_;
    }

    template <class V>
    bool operator!=(iterator_base<V> const& i) const noexcept {
      return element_ != i.element_;
    }

   private:
    intrusive_ptr<link_element> element_;

    explicit iterator_base(link_element* p) noexcept
      : element_{p} {
    }
  };

  using value_type    = T;
  using reference     = T&;
  using pointer       = T*;
  using const_pointer = const T*;

  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  using iterator               = iterator_base<T>;
  using const_iterator         = iterator_base<T const>;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  stable_list() {
    init();
  }

  ~stable_list() {
    destroy();
  }

  stable_list(stable_list const& l) {
    init();
    insert(end(), l.begin(), l.end());
  }

  stable_list(stable_list&& l)
    : head_{std::move(l.head_)}
    , tail_{std::move(l.tail_)}
    , elements_{l.elements_} {
    l.init();
  }

  stable_list(std::initializer_list<value_type> l) {
    init();
    insert(end(), l.begin(), l.end());
  }

  template <class Iterator>
  stable_list(Iterator ibegin, Iterator iend) {
    init();
    insert(end(), ibegin, iend);
  }

  explicit stable_list(size_type count, value_type const& value) {
    init();
    insert(end(), count, value);
  }

  explicit stable_list(size_type count) {
    init();
    insert(end(), count, value_type{});
  }

  stable_list& operator=(stable_list const& l) {
    if (this != &l) {
      clear();
      insert(end(), l.begin(), l.end());
    }
    return *this;
  }

  stable_list& operator=(stable_list&& l) {
    destroy();
    head_     = std::move(l.head_);
    tail_     = std::move(l.tail_);
    elements_ = l.elements_;
    l.init();
    return *this;
  }

  iterator begin() noexcept {
    return iterator{head_->next};
  }

  iterator end() noexcept {
    return iterator{tail_};
  }

  const_iterator begin() const noexcept {
    return const_iterator{head_->next};
  }

  const_iterator end() const noexcept {
    return const_iterator{tail_};
  }

  const_iterator cbegin() const noexcept {
    return const_iterator{head_->next};
  }

  const_iterator cend() const noexcept {
    return const_iterator{tail_};
  }

  reverse_iterator rbegin() noexcept {
    return reverse_iterator{end()};
  }

  reverse_iterator rend() noexcept {
    return reverse_iterator{begin()};
  }

  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator{cend()};
  }

  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator{cbegin()};
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator{cend()};
  }

  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator{cbegin()};
  }

  reference front() noexcept {
    return *begin();
  }

  reference back() noexcept {
    return *rbegin();
  }

  value_type const& front() const noexcept {
    return *cbegin();
  }

  value_type const& back() const noexcept {
    return *crbegin();
  }

  [[nodiscard]] bool empty() const noexcept {
    return cbegin() == cend();
  }

  void clear() noexcept {
    erase(begin(), end());
  }

  void push_front(value_type const& value) {
    insert(begin(), value);
  }

  void push_front(value_type&& value) {
    insert(begin(), std::move(value));
  }

  void push_back(value_type const& value) {
    insert(end(), value);
  }

  void push_back(value_type&& value) {
    insert(end(), std::move(value));
  }

  template <class... Args>
  reference emplace_front(Args&&... args) {
    return *emplace(begin(), std::forward<Args>(args)...);
  }

  template <class... Args>
  reference emplace_back(Args&&... args) {
    return *emplace(end(), std::forward<Args>(args)...);
  }

  void pop_front() noexcept {
    head_->next       = head_->next->next;
    head_->next->prev = head_;
    --elements_;
  }

  void pop_back() noexcept {
    tail_->prev       = tail_->prev->prev;
    tail_->prev->next = tail_;
    --elements_;
  }

  iterator insert(iterator const& pos, value_type const& value) {
    return iterator{make_link(pos.element, value)};
  }

  iterator insert(iterator const& pos, value_type&& value) {
    return iterator{make_link(pos.element, std::move(value))};
  }

  template <class Iterator>
  iterator insert(iterator const& pos, Iterator ibegin, Iterator iend) {
    iterator iter{end()};
    while (ibegin != iend) {
      iterator tmp{insert(pos, *ibegin++)};
      if (iter == end()) {
        iter = std::move(tmp);
      }
    }
    return iter;
  }

  iterator insert(iterator const& pos, std::initializer_list<value_type> l) {
    return insert(pos, l.begin(), l.end());
  }

  iterator insert(
      iterator const& pos, size_type count, value_type const& value) {
    iterator iter{end()};
    for (size_type i = 0; i < count; ++i) {
      iterator tmp{insert(pos, value)};
      if (iter == end()) {
        iter = std::move(tmp);
      }
    }
    return iter;
  }

  template <class... Args>
  iterator emplace(iterator const& pos, Args&&... args) {
    return iterator{make_link(pos.element, std::forward<Args>(args)...)};
  }

  void append(value_type const& value) {
    insert(end(), value);
  }

  void append(value_type&& value) {
    insert(end(), std::move(value));
  }

  template <class Iterator>
  void append(Iterator ibegin, Iterator iend) {
    insert(end(), ibegin, iend);
  }

  void append(std::initializer_list<value_type> l) {
    insert(end(), std::move(l));
  }

  void append(size_type count, value_type const& value) {
    insert(end(), count, value);
  }

  void assign(size_type count, value_type const& value) {
    clear();
    append(count, value);
  }

  template <class Iterator>
  void assign(Iterator ibegin, Iterator iend) {
    clear();
    append(ibegin, iend);
  }

  void assign(std::initializer_list<value_type> l) {
    clear();
    append(std::move(l));
  }

  void resize(size_type count) {
    resize(count, value_type{});
  }

  void resize(size_type count, value_type const& value) {
    size_type cursize = size();
    if (count > cursize) {
      for (size_type i = cursize; i < count; ++i) {
        push_back(value);
      }
    } else {
      for (size_type i = count; i < cursize; ++i) {
        pop_back();
      }
    }
  }

  [[nodiscard]] size_type size() const noexcept {
    return elements_;
  }

  [[nodiscard]] size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max();
  }

  iterator erase(iterator const& pos) noexcept {
    pos.element->prev->next = pos.element->next;
    pos.element->next->prev = pos.element->prev;
    --elements_;
    return iterator{pos.element->next};
  }

  iterator erase(iterator const& first, iterator const& last) noexcept {
    auto link = first.element;
    while (link != last.element) {
      auto next  = link->next;
      link->prev = first.element->prev;
      link->next = last.element;
      --elements_;
      link = std::move(next);
    }

    first.element->prev->next = last.element;
    last.element->prev        = first.element->prev;
    return last;
  }

  void remove(value_type const& value) noexcept {
    for (auto itr = begin(); itr != end(); ++itr) {
      if (*itr == value) {
        erase(itr);
      }
    }
  }

  template <class Predicate>
  void remove_if(Predicate const& pred) {
    for (auto itr = begin(); itr != end(); ++itr) {
      if (pred(*itr)) {
        erase(itr);
      }
    }
  }

  void swap(stable_list& other) noexcept {
    if (this != &other) {
      intrusive_ptr<link_element> tmp_head{std::move(head_)};
      intrusive_ptr<link_element> tmp_tail{std::move(tail_)};
      std::size_t                 tmp_elements{elements_};

      head_     = std::move(other.head_);
      tail_     = std::move(other.tail_);
      elements_ = other.elements_;

      other.head_     = std::move(tmp_head);
      other.tail_     = std::move(tmp_tail);
      other.elements_ = tmp_elements;
    }
  }

 private:
  void init() {
    head_       = new link_element;
    tail_       = new link_element;
    head_->next = tail_;
    tail_->prev = head_;
    elements_   = 0;
  }

  void destroy() {
    clear();
    head_->next = nullptr;
    tail_->prev = nullptr;
  }

  template <class... Args>
  link_element* make_link(link_element* l, Args&&... args) {
    intrusive_ptr<link_element> link{new link_element};
    link->construct(std::forward<Args>(args)...);
    link->prev       = l->prev;
    link->next       = l;
    link->prev->next = link;
    link->next->prev = link;
    ++elements_;
    return link;
  }
};
}  // namespace fmpi
#endif
