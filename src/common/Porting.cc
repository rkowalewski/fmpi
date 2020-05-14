#include <fmpi/common/Porting.hpp>
#include <fmpi/concurrency/OpenMP.hpp>

#include <thread>

namespace fmpi {

int get_num_user_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return std::thread::hardware_concurrency();
#endif
}

bool pinThreadToCore(std::thread& thread, int core_id) {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(core_id % std::thread::hardware_concurrency(), &cpuSet);
  auto const rc =
      pthread_setaffinity_np(thread.native_handle(), sizeof(cpuSet), &cpuSet);

  return rc == 0;
}
}  // namespace fmpi
