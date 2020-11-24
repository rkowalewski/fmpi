#ifndef FMPI_DETAIL_TAGS_HPP
#define FMPI_DETAIL_TAGS_HPP

#include <cstdint>

namespace fmpi {
namespace detail {
constexpr int32_t TAG_NEIGHBOR_ALLTOALL_CARTESIAN_BASE = 110400;
constexpr int32_t TAG_NEIGHBOR_ALLTOALL_GRAPH          = 110401;
constexpr int32_t TAG_ALLTOALL                         = 110435;
}  // namespace detail

}  // namespace fmpi

#endif
