#include <fmpi/concurrency/Dispatcher.hpp>

std::atomic_uint32_t fmpi::ScheduleHandle::last_id_ = 0;
