#ifndef FMPI_UTILS_HPP
#define FMPI_UTILS_HPP

#include <thread>
#include <type_traits>

namespace fmpi {

namespace detail {
template <typename E>
using enable_enum_t = typename std::enable_if<
    std::is_enum<E>::value,
    typename std::underlying_type<E>::type>::type;

}  // namespace detail

template <typename E>
constexpr inline detail::enable_enum_t<E> to_underlying(E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}

inline bool pinThreadToCore(std::thread& thread, int core_id) {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(core_id % std::thread::hardware_concurrency(), &cpuSet);
  auto const rc =
      pthread_setaffinity_np(thread.native_handle(), sizeof(cpuSet), &cpuSet);

  return rc == 0;
}

}  // namespace fmpi
#endif
