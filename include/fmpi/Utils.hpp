#ifndef FMPI_UTILS_HPP
#define FMPI_UTILS_HPP

#include <type_traits>

namespace fmpi {

namespace detail {
template <typename E>
using enable_enum_t = typename std::enable_if<
    std::is_enum<E>::value,
    typename std::underlying_type<E>::type>::type;

}  // namespace detail

template <typename E>
constexpr inline detail::enable_enum_t<E> to_underlying(E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}
}  // namespace fmpi
#endif
