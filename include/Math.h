#ifndef MATH_H__INCLUDED
#define MATH_H__INCLUDED

#include <Debug.h>

namespace a2a {

template <class T>
inline constexpr T mod(T a, T b)
{
  A2A_ASSERT(b > 0);
  return (a < 0) ? (a % b + b) : (a % b);
}

}  // namespace a2a

#endif
