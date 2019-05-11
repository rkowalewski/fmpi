#ifndef BENCHMARK_H__INCLUDED
#define BENCHMARK_H__INCLUDED

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include <Timer.h>

struct StringDoublePair : std::pair<std::string, double> {
  using std::pair<std::string, double>::pair;
};

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs);
std::ostream& operator<<(std::ostream& os, StringDoublePair const& p);

template <class InputIt, class OutputIt, class F>
auto run_algorithm(
    F&& f, InputIt begin, OutputIt out, int blocksize, MPI_Comm comm)
{
  auto start = ChronoClockNow();
  f(begin, out, blocksize, comm);
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
  else {
    return -1.0;
  }
}

void print_env();

#endif
