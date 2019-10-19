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
  bool         check{false};
} Params;

bool process(int /*argc*/, char* argv[], ::mpi::Context const&, Params&);

}  // namespace benchmark
}  // namespace fmpi
#endif
