#ifndef FMPI_OPENMP_HPP
#define FMPI_OPENMP_HPP
#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int         omp_int_t;
constexpr omp_int_t omp_get_thread_num() {
  return 0;
}
constexpr omp_int_t omp_get_max_threads() {
  return 1;
}
#endif
#endif
