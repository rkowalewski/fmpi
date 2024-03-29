#ifndef FMPI_UTIL_MATH_HPP
#define FMPI_UTIL_MATH_HPP

#include <climits>
#include <cmath>
#include <fmpi/detail/Assert.hpp>
#include <type_traits>

namespace fmpi {

#if 0
namespace detail {

template <class T>
inline constexpr auto mod(T a, T b, std::true_type /*unused*/) -> T
{
  static_assert(std::is_signed<T>::value, "only signed types allowed");
  auto const zero = T{0};
  FMPI_ASSERT(b > zero);
  return (a < zero) ? (a % b + b) : (a % b);
}

template <class T>
inline constexpr auto mod(T a, T b, std::false_type /*unused*/) -> T
{
  static_assert(std::is_unsigned<T>::value, "only unsigned types allowed");
  return a % b;
}

}  // namespace detail

template <class T>
inline constexpr auto mod(T a, T b) -> T
{
  static_assert(std::is_integral<T>::value, "only integer types supported");

  return detail::mod(a, b, typename std::is_signed<T>::type{});
}
#endif

template <class T>
inline constexpr auto log(T base, T x) -> double {
  return std::log(x) / std::log(base);
}

template <class T>
inline constexpr auto isPow2(T v) {
  static_assert(
      std::is_integral<T>::value && std::is_unsigned<T>::value,
      "only unsigned integer types supported");

  return (v & (v - 1)) == 0;
}

template <class T>
inline constexpr auto abs(T v) {
  static_assert(
      std::is_integral<T>::value && std::is_signed<T>::value,
      "only signed integer types supported");

  auto const mask = v >> (sizeof(T) * CHAR_BIT - 1);
  return (v + mask) ^ mask;
}

}  // namespace fmpi

#endif
