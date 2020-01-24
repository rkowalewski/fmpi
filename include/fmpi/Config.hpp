#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <cstddef>

namespace fmpi {

constexpr const char TOTAL[]         = "Ttotal";
constexpr const char MERGE[]         = "Tcomp";
constexpr const char COMMUNICATION[] = "Tcomm";

constexpr int EXCH_TAG_RING  = 110435;
constexpr int EXCH_TAG_BRUCK = 110436;

// Cache Configuration
constexpr std::size_t kCacheLineSize      = 64;
constexpr std::size_t kCacheLineAlignment = 64;

#if 0
class Config {
  int main_core{};
  int dispatcher_core{};
  int scheduler_core{};
  int comp_core{};
};
#endif

}  // namespace fmpi
#endif
