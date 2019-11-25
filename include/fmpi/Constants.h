#ifndef FMPI_CONSTANTS_H
#define FMPI_CONSTANTS_H

#include <cstddef>

namespace fmpi {
static constexpr const char TOTAL[] = "Ttotal";
static constexpr const char MERGE[]         = "Tcomp";
static constexpr const char COMMUNICATION[] = "Tcomm";


static constexpr int EXCH_TAG_RING  = 110435;
static constexpr int EXCH_TAG_BRUCK = 110436;


extern const std::size_t CACHELEVEL2_SIZE;

}  // namespace fmpi
#endif
