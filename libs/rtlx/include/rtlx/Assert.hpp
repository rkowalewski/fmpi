#ifndef RTLX_ASSERT_HPP
#define RTLX_ASSERT_HPP
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

// Stolen from Abseil
//
// RTLX_PREDICT_TRUE, RTLX_PREDICT_FALSE
//
// Enables the compiler to prioritize compilation using static analysis for
// likely paths within a boolean branch.
//
// Example:
//
//   if (RTLX_PREDICT_TRUE(expression)) {
//     return result;                        // Faster if more likely
//   } else {
//     return 0;
//   }
//
// Compilers can use the information that a certain branch is not likely to be
// taken (for instance, a CHECK failure) to optimize for the common case in
// the absence of better information (ie. compiling gcc with
// `-fprofile-arcs`).
//
// Recommendation: Modern CPUs dynamically predict branch execution paths,
// typically with accuracy greater than 97%. As a result, annotating every
// branch in a codebase is likely counterproductive; however, annotating
// specific branches that are both hot and consistently mispredicted is likely
// to yield performance improvements.
#if (defined(__GNUC__) && !defined(__clang__))
#define RTLX_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define RTLX_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define RTLX_PREDICT_FALSE(x) (x)
#define RTLX_PREDICT_TRUE(x) (x)
#endif

#if defined(NDEBUG)
#define RTLX_ASSERT(expr) \
  (false ? static_cast<void>(expr) : static_cast<void>(0))
#define RTLX_ASSERT_RETURNS(expr, ret) \
  (true ? static_cast<void>(expr) : static_cast<void>(0))
#else
#include <exception>
#define RTLX_ASSERT(expr)                                  \
  (RTLX_PREDICT_TRUE((expr)) ? static_cast<void>(0) : [] { \
    throw std::runtime_error{#expr};                       \
  }())  // NOLINT
/* #define RTLX_ASSERT_RETURNS(expr, ret) RTLX_ASSERT(((expr) == (ret))) */
#endif

#ifdef NDEBUG
#define RTLX_NOEXCEPT noexcept
#else
#define RTLX_NOEXCEPT
#endif

#endif
