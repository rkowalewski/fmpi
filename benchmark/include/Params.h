#ifndef PARAMS_H
#define PARAMS_H

#include <fmpi/mpi/Environment.h>

#include <iosfwd>
#include <vector>

namespace fmpi {
namespace benchmark {

constexpr size_t MINSZ = 32;
constexpr size_t MAXSZ = 128;

struct Params {
  unsigned int nhosts{};
#ifdef NDEBUG
  unsigned int niters{10};
#else
  unsigned int niters{1};
#endif
  std::string pattern{};
  bool        check{false};

  std::vector<std::size_t> sizes{};
};

auto process(
    int /*argc*/,
    char*                 argv[],
    ::mpi::Context const& mpiCtx,
    struct Params&        params) -> bool;

void printBenchmarkPreamble(
    std::ostream&      os,
    const std::string& prefix,
    const char*        delim = "\n");

}  // namespace benchmark
}  // namespace fmpi
#endif
