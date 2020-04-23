#ifndef FMPI_CONSTANTS_HPP
#define FMPI_CONSTANTS_HPP

#include <cstddef>
#include <string_view>

#include <fmpi/common/Porting.hpp>

namespace fmpi {

constexpr auto TOTAL         = std::string_view{"Ttotal"};
constexpr auto COMPUTATION   = std::string_view{"Tcomp"};
constexpr auto COMMUNICATION = std::string_view{"Tcomm"};
constexpr auto N_COMM_ROUNDS = std::string_view{"Ncomm_rounds"};

constexpr int EXCH_TAG_RING  = 110435;
constexpr int EXCH_TAG_BRUCK = 110436;

constexpr std::size_t kContainerStackSize      = 1024 * 512;
constexpr std::size_t kMaxContiguousBufferSize = 1024 * 512;

constexpr std::size_t kCachellineAlignment =
    std::hardware_destructive_interference_size;

constexpr std::size_t kCacheLineSize =
    std::hardware_constructive_interference_size;

}  // namespace fmpi
#endif
