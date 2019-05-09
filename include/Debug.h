#ifndef MACRO_H__INCLUDED
#define MACRO_H__INCLUDED
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

#ifdef NDEBUG
#define ASSERT(x)    \
  do {               \
    (void)sizeof(x); \
  } while (0)
#else
#include <cassert>
#define ASSERT(x) assert(x)
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

#endif
