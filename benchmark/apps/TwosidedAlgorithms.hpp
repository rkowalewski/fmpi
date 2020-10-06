#ifndef TWOSIDEDALGORITHMS_HPP  // NOLINT
#define TWOSIDEDALGORITHMS_HPP  // NOLINT

#include <fmpi/AlltoAll.hpp>
//#include <fmpi/Bruck.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <functional>
#include <regex>
#include <rtlx/Timer.hpp>
#include <rtlx/UnorderedMap.hpp>

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

void write_csv_header(std::ostream& os);

void write_csv_line(
    std::ostream&      os,
    Measurement const& params,
    std::pair<
        typename fmpi::TraceStore::key_type,
        typename fmpi::TraceStore::mapped_type> const& entry);

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
  using duration = typename rtlx::steady_timer::duration;

  duration d{};
  {
    rtlx::steady_timer t{d};
    f(begin, out, blocksize, comm, std::forward<Computation>(op));
  }
  return d;
}

template <
    class RandomAccessIterator1,
    class RandomAccessIterator2,
    class Callback>
auto algorithm_list(std::string const& pattern, mpi::Context const& ctx)
    -> std::unordered_map<
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
      algorithms{
          std::make_pair(
              "AlltoAll",
              fmpi::mpi_alltoall<
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback>),
#if 0
          std::make_pair(
              "RingWaitall4",
              fmpi::ring_waitall<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitall8",
              fmpi::ring_waitall<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitall16",
              fmpi::ring_waitall<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitall4",
              fmpi::ring_waitall<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitall8",
              fmpi::ring_waitall<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitall16",
              fmpi::ring_waitall<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  16>),
          std::make_pair(
              "RingWaitsome4",
              fmpi::ring_waitsome<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitsome8",
              fmpi::ring_waitsome<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitsome16",
              fmpi::ring_waitsome<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitsome4",
              fmpi::ring_waitsome<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitsome8",
              fmpi::ring_waitsome<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitsome16",
              fmpi::ring_waitsome<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  16>),
#endif
          std::make_pair(
              "RingWaitsomeOverlap4",
              fmpi::ring_waitsome_overlap<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "RingWaitsomeOverlap8",
              fmpi::ring_waitsome_overlap<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "RingWaitsomeOverlap16",
              fmpi::ring_waitsome_overlap<
                  fmpi::FlatHandshake,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  16>),
          std::make_pair(
              "OneFactorWaitsomeOverlap4",
              fmpi::ring_waitsome_overlap<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  4>),
          std::make_pair(
              "OneFactorWaitsomeOverlap8",
              fmpi::ring_waitsome_overlap<
                  fmpi::OneFactor,
                  RandomAccessIterator1,
                  RandomAccessIterator2,
                  Callback,
                  8>),
          std::make_pair(
              "OneFactorWaitsomeOverlap16",
              fmpi::ring_waitsome_overlap<
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
              "Bruck_interleave_dispatch",
              fmpi::bruck_interleave_dispatch<
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

  if (!pattern.empty()) {
    // remove algorithms not matching a pattern
    auto const regex = std::regex(pattern);
    rtlx::erase_if(algorithms, [regex](auto const& entry) {
      return !std::regex_match(entry.first, regex);
    });
  }

  if (!fmpi::isPow2(ctx.size())) {
    algorithms.erase("Bruck_Mod");
  }

  return algorithms;
}

template <class Iter1, class Iter2>
void validate(
    Iter1               first,
    Iter1               last,
    Iter2               expected,
    mpi::Context const& ctx,
    std::string const&  algo) {
  auto const is_equal = std::equal(first, last, expected);

  if (!is_equal) {
    std::ostringstream os;
    os << "[ERROR] [Rank " << ctx.rank() << "] " << algo
       << ": incorrect sequence";
    std::cerr << os.str();
    std::terminate();
  }
}

#endif
