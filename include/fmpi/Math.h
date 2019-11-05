#ifndef FMPI_MATH_H
#define FMPI_MATH_H

#include <rtlx/Assert.h>

#include <type_traits>

#include <climits>

namespace fmpi {

namespace detail {

template <class T>
inline constexpr T mod(T a, T b, std::true_type /*unused*/)
{
  static_assert(std::is_signed<T>::value, "only signed types allowed");
  auto const zero = T{0};
  RTLX_ASSERT(b > zero);
  return (a < zero) ? (a % b + b) : (a % b);
}

template <class T>
inline constexpr T mod(T a, T b, std::false_type /*unused*/)
{
  static_assert(std::is_unsigned<T>::value, "only unsigned types allowed");
  return a % b;
}
}  // namespace detail

template <class T>
inline constexpr T mod(T a, T b)
{
  static_assert(std::is_integral<T>::value, "only integer types supported");

  return detail::mod(a, b, typename std::is_signed<T>::type{});
}

template <class T>
inline constexpr auto isPow2(T v)
{
  static_assert(
      std::is_integral<T>::value && std::is_unsigned<T>::value,
      "only unsigned integer types supported");

  return (v & (v - 1)) == 0;
}

template <class T>
inline constexpr auto abs(T v)
{
  static_assert(
      std::is_integral<T>::value && std::is_signed<T>::value,
      "only signed integer types supported");

  auto const mask = v >> (sizeof(T) * CHAR_BIT - 1);
  return (v + mask) ^ mask;
}

}  // namespace fmpi

#endif
