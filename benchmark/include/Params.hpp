#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <fmpi/mpi/Environment.hpp>
#include <iosfwd>
#include <vector>

namespace fmpi {
namespace benchmark {

constexpr size_t MINSZ = 32;
constexpr size_t MAXSZ = 128;

enum class Progress : uint8_t
{
  Nonblocking = 0,
  Blocking
};

struct Params {
  unsigned int nhosts{};
#ifdef NDEBUG
  unsigned int niters{10};
#else
  unsigned int niters{1};
#endif
  std::string pattern{};
  bool        check{false};
  bool        blocking_progress{false};

  std::vector<std::size_t> sizes{};
};

auto process(
    int /*argc*/,
    char*                 argv[],
    ::mpi::Context const& mpiCtx,
    struct Params&        params) -> bool;

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim = "\n");

}  // namespace benchmark
}  // namespace fmpi
#endif
