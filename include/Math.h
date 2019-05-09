#ifndef MATH_H__INCLUDED
#define MATH_H__INCLUDED

#include <Debug.h>

template <class T>
inline constexpr T mod(T a, T b)
{
  ASSERT(b > 0);
  T ret = a % b;
  return (ret >= 0) ? (ret) : (ret + b);
}

#endif
