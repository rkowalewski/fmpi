#ifndef RTLX_TIMER_H
#define RTLX_TIMER_H

#include <chrono>

namespace rtlx {

template <
    bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct ChooseSteadyClock {
  using type = std::chrono::high_resolution_clock;
};

template <>
struct ChooseSteadyClock<false> {
  using type = std::chrono::steady_clock;
};

struct ChooseClockType {
  using type = ChooseSteadyClock<>::type;
};

inline auto ChronoClockNow() -> double
{
  using ClockType = ChooseClockType::type;
  using duration_t =
      std::chrono::duration<double, std::chrono::seconds::period>;
  return duration_t(ClockType::now().time_since_epoch()).count();
}
}  // namespace rtlx

#endif
