#ifndef FMPI_SPAN_HPP
#define FMPI_SPAN_HPP
#include <iosfwd>
#include <memory>

namespace fmpi {

#ifdef cpp_lib_byte
using byte = std::byte;
#else
using byte = unsigned char;
#endif

template <class T>
class Span {
  T*          data_ = nullptr;
  std::size_t size_ = 0;

 public:
  /// The type of value, including cv qualifiers
  using element_type = T;

  /// The type of value of each Span element
  using value_type = typename std::remove_const<T>::type;

  /// The type of integer used to index the Span
  using index_type = std::ptrdiff_t;

  /// A pointer to a Span element
  using pointer = T*;

  /// A reference to a Span element
  using reference = T&;

  /// The iterator used by the container
  using iterator = pointer;

  /// The const pointer used by the container
  using const_pointer = T const*;

  /// The const reference used by the container
  using const_reference = T const&;

  /// The const iterator used by the container
  using const_iterator = const_pointer;

  /// Constructor
  Span() = default;

  /// Constructor
  Span(Span const&) = default;

  /// Assignment
  Span& operator=(Span const&) = default;

  /** Constructor

      @param data A pointer to the beginning of the range of elements

      @param size The number of elements pointed to by `data`
  */
  Span(T* data, std::size_t size) noexcept
    : data_(data)
    , size_(size) {
  }

  template <class CharT, class Traits, class Allocator>
  explicit Span(std::basic_string<CharT, Traits, Allocator>& s) noexcept
    : data_(&s[0])
    , size_(s.size()) {
  }

  template <class CharT, class Traits, class Allocator>
  explicit Span(std::basic_string<CharT, Traits, Allocator> const& s) noexcept
    : data_(s.data())
    , size_(s.size()) {
  }

  template <class CharT, class Traits, class Allocator>
  Span& operator=(std::basic_string<CharT, Traits, Allocator>& s) noexcept {
    data_ = &s[0];
    size_ = s.size();
    return *this;
  }

  template <class CharT, class Traits, class Allocator>
  Span& operator=(
      std::basic_string<CharT, Traits, Allocator> const& s) noexcept {
    data_ = s.data();
    size_ = s.size();
    return *this;
  }

  /// Returns `true` if the Span is empty
  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  /// Returns a pointer to the beginning of the Span
  [[nodiscard]] element_type* data() const {
    return data_;
  }

  /// Returns the number of elements in the Span
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return size() * sizeof(element_type);
  }

  /// Returns an iterator to the beginning of the Span
  [[nodiscard]] const_iterator begin() const noexcept {
    return data_;
  }

  /// Returns an iterator to the beginning of the Span
  [[nodiscard]] const_iterator cbegin() const noexcept {
    return data_;
  }

  /// Returns an iterator to one past the end of the Span
  [[nodiscard]] const_iterator end() const noexcept {
    return data_ + size_;
  }

  /// Returns an iterator to one past the end of the Span
  [[nodiscard]] const_iterator cend() const noexcept {
    return data_ + size_;
  }
};

template <class T>
Span<byte> as_bytes(Span<T> s) noexcept {
  return {reinterpret_cast<const byte*>(s.data()), s.size_bytes()};
}

template <class T>
Span<byte> as_writable_bytes(Span<T> s) noexcept {
  return {reinterpret_cast<byte*>(s.data()), s.size_bytes()};
}

template <class Iter>
auto make_span(Iter begin, std::size_t count) noexcept {
  return Span<typename std::iterator_traits<Iter>::value_type>(
      &*begin, count);
}

#if 0
template <class T>
std::ostream& operator<<(std::ostream& os, Span<T> const& span) {
  FMPI_ASSERT(false);
  auto p = static_cast<const void*>(span.data());
  os << std::string("{ 'data': ") << p << std::string(", 'size': ")
     << span.size() << " }";
  return os;
}
#endif

}  // namespace fmpi
#endif
