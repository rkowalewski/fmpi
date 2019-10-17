#ifndef FMPI__BENCHMARK__PARAMS_H
#define FMPI__BENCHMARK__PARAMS_H

#include <array>

#include <fmpi/mpi/Environment.h>

namespace fmpi::benchmark {

constexpr size_t MINSZ = 128;
constexpr size_t MAXSZ = 256;

typedef struct Params {
  std::size_t minblocksize{MINSZ};
  std::size_t maxblocksize{MAXSZ};
  unsigned int         nhosts{};
} Params;

bool process(int, char* argv[], ::mpi::MpiCommCtx const&, Params& /* inout */);

}  // namespace fmpi::benchmark
#endif
