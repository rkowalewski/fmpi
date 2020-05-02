#ifndef FMPI_MEMORY_DETAIL_POINTERARITHMETIC_HPP
#define FMPI_MEMORY_DETAIL_POINTERARITHMETIC_HPP

#include <cstddef>
#include <cstdint>

namespace fmpi {

namespace detail {

inline std::size_t alignForwardAdjustment(
    const void* address, std::size_t alignment) {
  std::size_t adjustment =
      alignment - (reinterpret_cast<std::uintptr_t>(address) &
                   static_cast<std::uintptr_t>(alignment - 1));

  if (adjustment == alignment) return 0;  // already aligned

  return adjustment;
}

template <class T>
inline bool isAligned(const T* address) {
  return alignForwardAdjustment(address, alignof(T)) == 0;
}

inline bool isAligned(const void* address, std::size_t alignment) {
  return alignForwardAdjustment(address, alignment) == 0;
}

inline void* add(void* p, size_t x) {
  return (void*)(reinterpret_cast<std::uintptr_t>(p) + x);
}

inline const void* add(const void* p, size_t x) {
  return (const void*)(reinterpret_cast<std::uintptr_t>(p) + x);
}

inline void* sub(void* p, size_t x) {
  return (void*)(reinterpret_cast<std::uintptr_t>(p) - x);
}

inline const void* sub(const void* p, size_t x) {
  return (const void*)(reinterpret_cast<std::uintptr_t>(p) - x);
}

}  // namespace detail
}  // namespace fmpi
#endif
