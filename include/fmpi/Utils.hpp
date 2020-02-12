#ifndef FMPI_UTILS_HPP
#define FMPI_UTILS_HPP

#include <thread>

namespace fmpi {

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
