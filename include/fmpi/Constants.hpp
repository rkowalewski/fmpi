#ifndef FMPI_CONSTANTS_HPP
#define FMPI_CONSTANTS_HPP

#include <cstddef>

namespace fmpi {

static constexpr const char TOTAL[]         = "Ttotal";
static constexpr const char MERGE[]         = "Tcomp";
static constexpr const char COMMUNICATION[] = "Tcomm";

static constexpr int EXCH_TAG_RING  = 110435;
static constexpr int EXCH_TAG_BRUCK = 110436;

extern const std::size_t CACHELEVEL2_SIZE;

static constexpr std::size_t CACHELINE_LENGTH = 64;
static constexpr std::size_t CACHE_ALIGNMENT  = 64;

}  // namespace fmpi
#endif
