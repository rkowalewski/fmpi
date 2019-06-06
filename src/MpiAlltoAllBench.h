#ifndef MPI_ALLTOALL_BENCH_H__INCLUDED
#define MPI_ALLTOALL_BENCH_H__INCLUDED
#include <mpi.h>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <Timer.h>
#include <Trace.h>

struct StringDoublePair : std::pair<std::string, double> {
  using std::pair<std::string, double>::pair;
};

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs);
std::ostream& operator<<(std::ostream& os, StringDoublePair const& p);

using merge_t = std::function<void(void*, void*, void*, void*, void*)>;

struct Params {
  size_t nhosts;
  size_t nprocs;
  int    me;
  size_t step;
  size_t nbytes;
  size_t blocksize;
};

void printMeasurementHeader(std::ostream& os);
void printTraceHeader(std::ostream& os);

void printMeasurementCsvLine(
    std::ostream& os, Params params, std::string algorithm, double time);

void printTraceCsvLine(std::ostream& os, TimeTrace const& trace);

template <class InputIt, class OutputIt, class CommAlgo, class Merger>
auto run_algorithm(
    CommAlgo&& f,
    InputIt    begin,
    OutputIt   out,
    int        blocksize,
    MPI_Comm   comm,
    Merger&&   op)
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
