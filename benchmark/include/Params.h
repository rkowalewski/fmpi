#ifndef PARAMS_H
#define PARAMS_H

#include <array>

#include <fmpi/mpi/Environment.h>

#include <iosfwd>

namespace fmpi {
namespace benchmark {

constexpr size_t MINSZ = 32;
constexpr size_t MAXSZ = 128;

struct Params {
  std::size_t  minblocksize{MINSZ};
  std::size_t  maxblocksize{MAXSZ};
  unsigned int nhosts{};
#ifdef NDEBUG
  unsigned int niters{10};
#else
  unsigned int niters{1};
#endif
  std::string pattern{};
  bool        check{false};
};

bool process(
    int /*argc*/,
    char*                 argv[],
    ::mpi::Context const& mpiCtx,
    struct Params&        params);

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim = "\n");

}  // namespace benchmark
}  // namespace fmpi
#endif
