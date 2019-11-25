#ifndef BENCHMARK__TWOSIDED_ALGORITHMS_H
#define BENCHMARK__TWOSIDED_ALGORITHMS_H

#include <rtlx/Timer.h>
#include <rtlx/Trace.h>

#include <functional>

struct Measurement {
  size_t nhosts;
  size_t nprocs;
  size_t nthreads;
  int    me;

  size_t step;
  size_t nbytes;
  size_t blocksize;
  size_t iter;

  std::string algorithm;
};

void printMeasurementHeader(std::ostream& os);

void printMeasurementCsvLine(
    std::ostream&                                                     os,
    Measurement const&                                                params,
    std::unordered_map<std::string, std::variant<double, int>> const& traces);

template <
    class InputIt,
    class OutputIt,
    class Communication,
    class Computation>
auto run_algorithm(
    Communication&&     f,
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& comm,
    Computation&&       op)
{
  auto start = rtlx::ChronoClockNow();
  f(begin, out, blocksize, comm, std::forward<Computation>(op));
  return rtlx::ChronoClockNow() - start;
}

template <class Iterator, class Callback>
using fmpi_algrithm_t = std::function<void(
    Iterator, Iterator, int, mpi::Context const&, Callback)>;

#endif
