#ifndef DEBUG_H
#define DEBUG_H
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

// Stolen from Abseil
//
// A2A_PREDICT_TRUE, A2A_PREDICT_FALSE
//
// Enables the compiler to prioritize compilation using static analysis for
// likely paths within a boolean branch.
//
// Example:
//
//   if (A2A_PREDICT_TRUE(expression)) {
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
#define A2A_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define A2A_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define A2A_PREDICT_FALSE(x) (x)
#define A2A_PREDICT_TRUE(x) (x)
#endif

#if defined(NDEBUG)
#define A2A_ASSERT(expr) \
  (false ? static_cast<void>(expr) : static_cast<void>(0))

#define A2A_ASSERT_RETURNS(expr, ret) \
  (true ? static_cast<void>(expr) : static_cast<void>(0))
#else
#define A2A_ASSERT(expr)                           \
  (A2A_PREDICT_TRUE((expr)) ? static_cast<void>(0) \
                            : [] { assert(false && #expr); }())  // NOLINT
#define A2A_ASSERT_RETURNS(expr, ret) A2A_ASSERT((expr) == (ret))
#endif

template <class InputIt>
void printVector(InputIt begin, InputIt end, int me)
{
  using value_t = typename std::iterator_traits<InputIt>::value_type;

  std::ostringstream os;
  os << "rank " << me << ": ";
  std::copy(begin, end, std::ostream_iterator<value_t>(os, " "));
  os << "\n";
  std::cout << os.str();
}

template <class InputIt>
auto tokenizeRange(InputIt begin, InputIt end)
{
  std::ostringstream os;
  using value_t = typename std::iterator_traits<InputIt>::value_type;
  std::copy(begin, end, std::ostream_iterator<value_t>(os, " "));
  return os.str();
}

#define P(x)
#define PRANGE(a, b)

#ifndef NDEBUG
#ifdef A2A_ENABLE_LOGGING
#undef P
// NOLINT
#define P(x)                                                                 \
  do {                                                                       \
    std::ostringstream os;                                                   \
    os << "-- [ " << __func__ << ":" << __LINE__ << " ] " << x << std::endl; \
    std::cout << os.str();                                                   \
  } while (0)
// DISABLE_NOLINT

#undef PRANGE
#define PRANGE(a, b)                                                        \
  do {                                                                      \
    std::ostringstream os;                                                  \
    using value_t = typename std::iterator_traits<decltype(a)>::value_type; \
    os << "-- [ " << __func__ << ":" << __LINE__ << " ] ";                  \
    std::copy(a, b, std::ostream_iterator<value_t>(os, " "));               \
    os << std::endl;                                                        \
    std::cout << os.str();                                                  \
  } while (0)
#endif
#endif

#endif
