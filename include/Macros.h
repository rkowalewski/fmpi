#ifndef MACRO_H__INCLUDED
#define MACRO_H__INCLUDED
#ifdef NDEBUG
#define ASSERT(x)    \
  do {               \
    (void)sizeof(x); \
  } while (0)
#else
#include <cassert>
#define ASSERT(x) assert(x)
#endif
#endif
