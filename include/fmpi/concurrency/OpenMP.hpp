#ifndef FMPI_CONCURRENCY_OPENMP_HPP
#define FMPI_CONCURRENCY_OPENMP_HPP
#if defined(_OPENMP)
#include <omp.h>
#else
typedef int         omp_int_t;
omp_int_t omp_get_thread_num() {
  return 0;
}
omp_int_t omp_get_max_threads() {
  return 1;
}
#endif
#endif
