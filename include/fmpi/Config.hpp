#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <cstddef>
#include <fmpi/common/Porting.hpp>
#include <string_view>

#define FMPI_IMPL_IN_CONFIG_HPP
#include "config_impl.hpp"
#undef FMPI_IMPL_IN_CONFIG_HPP

#define FMPI_LOG_PREFIX "fmpi"

#ifdef FMPI_DEBUG_ASSERT
#define FMPI_NOEXCEPT
#else
#define FMPI_NOEXCEPT noexcept
#endif

namespace fmpi {

constexpr bool kEnableTrace = FMPI_ENABLE_TRACE;

// Cache Configuration
constexpr std::size_t kCacheSizeL2 = FMPI_CACHELEVEL2_SIZE;
constexpr std::size_t kCacheSizeL3 = FMPI_CACHELEVEL3_SIZE;
constexpr std::size_t kCacheAlignment =
    std::hardware_destructive_interference_size;
constexpr std::size_t kCacheLineSize =
    std::hardware_constructive_interference_size;

// constexpr auto kTotalTime = std::string_view{"Ttotal"};

// constexpr auto kComputationTime = std::string_view{"Tcomp"};
// Time to perform communication (for example in the dispatcher)
constexpr auto kCommunicationTime = std::string_view{"Tcomm.time"};
// Time to initiate non-blocking communication
constexpr auto kScheduleTime = std::string_view{"Tcomm.schedule"};
// Idle time between communication and compute (e.g., in a blocking
// wait)
constexpr auto kIdle = std::string_view{"Tcomm.idle"};
// constexpr auto kCommRounds        = std::string_view{"Tcomm.iterations"};

constexpr int kTagRing  = 110435;
constexpr int kTagBruck = 110436;

constexpr std::size_t kContainerStackSize = 1024 * 512;  // 512 KBytes

}  // namespace fmpi
#endif
