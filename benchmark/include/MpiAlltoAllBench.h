#ifndef MPI_ALLTOALL_BENCH_H__INCLUDED
#define MPI_ALLTOALL_BENCH_H__INCLUDED

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <fusion/mpi/Mpi.h>

struct StringDoublePair : std::pair<std::string, double> {
  using std::pair<std::string, double>::pair;
};

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs);
std::ostream& operator<<(std::ostream& os, StringDoublePair const& p);

template <class InputIterator, class OutputIterator>
using merge_t = std::function<void(
    std::vector<std::pair<InputIterator, InputIterator>>, OutputIterator)>;

struct Params {
  size_t nhosts;
  size_t nprocs;
  int    me;
  size_t step;
  size_t nbytes;
  size_t blocksize;
};

void printMeasurementHeader(std::ostream& os);

void printMeasurementCsvLine(
    std::ostream&                      os,
    Params                             m,
    const std::string&                 algorithm,
    std::tuple<double, double, double> times);

template <class InputIt, class OutputIt, class CommAlgo, class Merger>
auto run_algorithm(
    CommAlgo&&             f,
    InputIt                begin,
    OutputIt               out,
    int                    blocksize,
    mpi::MpiCommCtx const& comm,
    Merger&&               op)
{
  auto start = ChronoClockNow();
  f(begin, out, blocksize, comm, std::forward<Merger>(op));
  return ChronoClockNow() - start;
}

template <class T>
auto medianReduce(T myMedian, int root, MPI_Comm comm)
{
  int me, nr;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &nr);
  std::vector<double> meds;

  meds.reserve(nr);

  MPI_Gather(&myMedian, 1, MPI_DOUBLE, &meds[0], 1, MPI_DOUBLE, root, comm);

  if (me == root) {
    auto nth = &meds[0] + nr / 2;
    std::nth_element(&meds[0], nth, &meds[0] + nr);
    return *nth;
  }

  return T{};
}

void print_env();

#endif
