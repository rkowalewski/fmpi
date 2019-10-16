#ifndef MATH_H
#define MATH_H

#include <rtlx/Assert.h>

#include <type_traits>

#include <climits>

namespace fmpi {

template <class T>
inline constexpr T mod(T a, T b)
{
  RTLX_ASSERT(b > 0);
  return (a < 0) ? (a % b + b) : (a % b);
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
