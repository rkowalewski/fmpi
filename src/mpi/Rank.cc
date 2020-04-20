#include <fmpi/mpi/Rank.hpp>

namespace mpi {

std::ostream& operator<<(std::ostream& os, Rank const& p) {
  os << static_cast<int>(p);
  return os;
}

}  // namespace mpi
