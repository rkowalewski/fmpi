#include <fmpi/Constants.h>
#include <fmpi/Operation.h>

namespace fmpi {
auto isCacheLevel2Utilized(std::size_t nbytes) noexcept -> bool
{
  return nbytes >= CACHELEVEL2_SIZE * 0.75;
}
}  // namespace fmpi
