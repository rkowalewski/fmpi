#ifndef PARAMS_H
#define PARAMS_H

#include <array>

#include <fmpi/mpi/Environment.h>

namespace fmpi {
namespace benchmark {

constexpr size_t MINSZ = 128;
constexpr size_t MAXSZ = 256;

typedef struct Params {
  std::size_t  minblocksize{MINSZ};
  std::size_t  maxblocksize{MAXSZ};
  unsigned int nhosts{};
#ifdef NDEBUG
  unsigned int niters{10};
#else
  unsigned int niters{1};
#endif
  bool check{false};
} Params;

bool process(
    int /*argc*/,
    char* argv[],
    ::mpi::Context const& mpiCtx,
    Params& params);

}  // namespace benchmark
}  // namespace fmpi
#endif
