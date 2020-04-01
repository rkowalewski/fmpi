#ifndef FMPI_COMMON_PORTING_HPP
#define FMPI_COMMON_PORTING_HPP

#include <thread>

namespace fmpi {

int get_num_threads();
bool pinThreadToCore(std::thread& thread, int core_id);

}  // namespace fmpi

#endif
