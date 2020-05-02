#ifndef FMPI_MEMORY_STDALLOCATOR_HPP
#define FMPI_MEMORY_STDALLOCATOR_HPP
namespace fmpi {

struct memory_arena {
};

template <class T, bool threadSafe = false>
class StdAllocator {};
}  // namespace fmpi
#endif
