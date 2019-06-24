#ifndef MACRO_H__INCLUDED
#define MACRO_H__INCLUDED
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

#define A2A_UNUSED(x) \
  do {                \
    (void)(x);        \
  } while (0)

#ifdef NDEBUG

#define A2A_ASSERT(x) A2A_UNUSED(x)

#define A2A_ASSERT_RETURNS(x, ret) \
  A2A_UNUSED(x);                   \
  A2A_UNUSED(ret)
#else

#include <cassert>
#define A2A_ASSERT(x) assert(x)
#define A2A_ASSERT_RETURNS(x, ret) A2A_ASSERT((x) == ret)
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

#define P(x)

#ifndef NDEBUG
#ifdef A2A_ENABLE_LOGGING
#undef P
#define P(x)                  \
  do {                        \
    std::ostringstream os;    \
    os << "-- " << x << "\n"; \
    std::cout << os.str();    \
  } while (0)
#endif
#endif

#endif
