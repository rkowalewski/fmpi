#ifndef MATH_H__INCLUDED
#define MATH_H__INCLUDED

#include <Debug.h>
#include <type_traits>

namespace a2a {

template <class T>
inline constexpr T mod(T a, T b)
{
  A2A_ASSERT(b > 0);
  return (a < 0) ? (a % b + b) : (a % b);
}

template <class T>
inline constexpr auto isPow2(T v)
{
  static_assert(
      std::is_integral<T>::value && std::is_unsigned<T>::value,
      "only unsigned types supported");

  return (v & (v - 1)) == 0;
}

}  // namespace a2a

#endif
