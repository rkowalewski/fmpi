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
    class RandomAccessIterator1,
    class RandomAccessIterator2,
    class Callback>
using fmpi_algorithm_t = std::function<void(
    RandomAccessIterator1,
    RandomAccessIterator2,
    int,
    mpi::Context const&,
    Callback)>;

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

template <
    class RandomAccessIterator1,
    class RandomAccessIterator2,
    class Callback>
constexpr auto algorithm_list() -> std::unordered_map<
    std::string,
    fmpi_algorithm_t<
        RandomAccessIterator1,
        RandomAccessIterator2,
        Callback>> {
  std::unordered_map<
      std::string,
      fmpi_algorithm_t<
          RandomAccessIterator1,
          RandomAccessIterator2,
          Callback>>
      algorithms {
#if 0
        std::make_pair(
                     "AlltoAll",
                     fmpi::MpiAlltoAll<
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback>),
                 std::make_pair(
                     "RingWaitall4",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         4>),
                 std::make_pair(
                     "RingWaitall8",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         8>),
                 std::make_pair(
                     "RingWaitall16",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         16>),
                 std::make_pair(
                     "OneFactorWaitall4",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         4>),
                 std::make_pair(
                     "OneFactorWaitall8",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         8>),
                 std::make_pair(
                     "OneFactorWaitall16",
                     fmpi::scatteredPairwiseWaitall<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         16>),
                 std::make_pair(
                     "RingWaitsome4",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         4>),
                 std::make_pair(
                     "RingWaitsome8",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         8>),
                 std::make_pair(
                     "RingWaitsome16",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::FlatHandshake,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         16>),
                 std::make_pair(
                     "OneFactorWaitsome4",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         4>),
                 std::make_pair(
                     "OneFactorWaitsome8",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         8>),
                 std::make_pair(
                     "OneFactorWaitsome16",
                     fmpi::scatteredPairwiseWaitsome<
                         fmpi::OneFactor,
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback,
                         16>),
#endif

    std::make_pair(
        "RingWaitsomeOverlap4",
        fmpi::scatteredPairwiseWaitsomeOverlap<
            fmpi::FlatHandshake,
            RandomAccessIterator1,
            RandomAccessIterator2,
            Callback,
            4>),
        std::make_pair(
            "RingWaitsomeOverlap8",
            fmpi::scatteredPairwiseWaitsomeOverlap<
                fmpi::FlatHandshake,
                RandomAccessIterator1,
                RandomAccessIterator2,
                Callback,
                8>),
        std::make_pair(
            "RingWaitsomeOverlap16",
            fmpi::scatteredPairwiseWaitsomeOverlap<
                fmpi::FlatHandshake,
                RandomAccessIterator1,
                RandomAccessIterator2,
                Callback,
                16>),
        std::make_pair(
            "OneFactorWaitsomeOverlap4",
            fmpi::scatteredPairwiseWaitsomeOverlap<
                fmpi::OneFactor,
                RandomAccessIterator1,
                RandomAccessIterator2,
                Callback,
                4>),
        std::make_pair(
            "OneFactorWaitsomeOverlap8",
            fmpi::scatteredPairwiseWaitsomeOverlap<
                fmpi::OneFactor,
                RandomAccessIterator1,
                RandomAccessIterator2,
                Callback,
                8>),
        std::make_pair(
            "OneFactorWaitsomeOverlap16",
            fmpi::scatteredPairwiseWaitsomeOverlap<
                fmpi::OneFactor,
                RandomAccessIterator1,
                RandomAccessIterator2,
                Callback,
                16>),
#if 0
                 // Bruck Algorithms, first the original one, then a modified
                 // version which omits the last local rotation step
                 std::make_pair(
                     "Bruck",
                     fmpi::bruck<
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback>),
                 std::make_pair(
                     "Bruck_indexed",
                     fmpi::bruck_indexed<
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback>),
                 std::make_pair(
                     "Bruck_interleave",
                     fmpi::bruck_interleave<
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback>),
                 std::make_pair(
                     "Bruck_Mod",
                     fmpi::bruck_mod<
                         RandomAccessIterator1,
                         RandomAccessIterator2,
                         Callback>)
#endif
  };

  return algorithms;
}

#endif
