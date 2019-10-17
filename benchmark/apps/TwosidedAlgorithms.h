#ifndef BENCHMARK__TWOSIDED_ALGORITHMS_H
#define BENCHMARK__TWOSIDED_ALGORITHMS_H

#include <rtlx/Timer.h>

struct Measurement {
  size_t nhosts;
  size_t nprocs;
  size_t step;
  size_t nbytes;
  size_t blocksize;
  int    me;

  std::string                        algorithm;
  std::tuple<double, double, double> times;
};

void printMeasurementHeader(std::ostream& os);
void printMeasurementCsvLine(std::ostream& os, Measurement const& m);

template <
    class InputIt,
    class OutputIt,
    class Communication,
    class Computation>
auto run_algorithm(
    Communication&&        f,
    InputIt                begin,
    OutputIt               out,
    int                    blocksize,
    mpi::MpiCommCtx const& comm,
    Computation&&          op)
{
  auto start = rtlx::ChronoClockNow();
  f(begin, out, blocksize, comm, std::forward<Computation>(op));
  return rtlx::ChronoClockNow() - start;
}

template <class Iterator, class Callback>
using fmpi_algrithm_t = std::function<void(
    Iterator, Iterator, int, mpi::MpiCommCtx const&, Callback)>;

#endif
