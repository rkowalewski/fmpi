#ifndef TWOSIDEDALGORITHMS_HPP  // NOLINT
#define TWOSIDEDALGORITHMS_HPP  // NOLINT

#include <functional>
#include <rtlx/Timer.hpp>
#include <rtlx/Trace.hpp>

#include <fmpi/AlltoAll.hpp>
#include <fmpi/Bruck.hpp>
#include <fmpi/mpi/Environment.hpp>

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
    std::ostream&      os,
    Measurement const& params,
    std::unordered_map<std::string, typename rtlx::TraceStore::value_t> const&
        traces);

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
    Computation&&       op) {
  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;

  duration d{};

  {
    timer t{d};
    f(begin, out, blocksize, comm, std::forward<Computation>(op));
  }

  return d;
}

template <class Iterator, class Callback>
using fmpi_algorithm_t = std::function<void(
    Iterator, Iterator, int, mpi::Context const&, Callback)>;

template <class Iterator, class Callback>
constexpr auto algorithm_list()
    -> std::unordered_map<std::string, fmpi_algorithm_t<Iterator, Callback>> {
  std::unordered_map<std::string, fmpi_algorithm_t<Iterator, Callback>>
      algorithms{
          std::make_pair(
              "AlltoAll", fmpi::MpiAlltoAll<Iterator, Iterator, Callback>),
          std::make_pair(
              "RingWaitall4",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitall8",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitall16",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitall4",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitall8",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitall16",
              fmpi::scatteredPairwiseWaitall<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          std::make_pair(
              "RingWaitsome4",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitsome8",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitsome16",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitsome4",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitsome8",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitsome16",
              fmpi::scatteredPairwiseWaitsome<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          std::make_pair(
              "RingWaitsomeOverlap4",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitsomeOverlap8",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitsomeOverlap16",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::FlatHandshake,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitsomeOverlap4",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitsomeOverlap8",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitsomeOverlap16",
              fmpi::scatteredPairwiseWaitsomeOverlap<
                  fmpi::OneFactor,
                  Iterator,
                  Iterator,
                  Callback,
                  16>),
          // Bruck Algorithms, first the original one, then a modified
          // version which omits the last local rotation step
          std::make_pair("Bruck", fmpi::bruck<Iterator, Iterator, Callback>),
          std::make_pair(
              "Bruck_indexed",
              fmpi::bruck_indexed<Iterator, Iterator, Callback>),
          std::make_pair(
              "Bruck_interleave",
              fmpi::bruck_interleave<Iterator, Iterator, Callback>),
          std::make_pair(
              "Bruck_Mod", fmpi::bruck_mod<Iterator, Iterator, Callback>)

      };

  return algorithms;
}

#endif
