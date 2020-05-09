#ifndef FMPI_CONSTANTS_HPP
#define FMPI_CONSTANTS_HPP

#include <cstddef>
#include <string_view>

#include <fmpi/common/Porting.hpp>

#define FMPI_IMPL_IN_CONFIG_HPP
#include "config_impl.hpp"
#undef FMPI_IMPL_IN_CONFIG_HPP

namespace fmpi {

constexpr auto kTotalTime         = std::string_view{"Ttotal"};
constexpr auto kComputationTime   = std::string_view{"Tcomp"};
constexpr auto kCommunicationTime = std::string_view{"Tcomm.time"};
constexpr auto kCommRounds        = std::string_view{"Tcomm.iterations"};

constexpr int kTagRing  = 110435;
constexpr int kTagBruck = 110436;

constexpr bool kEnableTrace = FMPI_ENABLE_TRACE;

constexpr std::size_t kContainerStackSize      = 1024 * 512;
constexpr std::size_t kMaxContiguousBufferSize = 1024 * 512;

// Cache Configuration
constexpr std::size_t kCacheSizeL2 = FMPI_CACHELEVEL2_SIZE;
constexpr std::size_t kCacheSizeL3 = FMPI_CACHELEVEL3_SIZE;
constexpr std::size_t kCacheAlignment =
    std::hardware_destructive_interference_size;
constexpr std::size_t kCacheLineSize =
    std::hardware_constructive_interference_size;

}  // namespace fmpi
#endif
