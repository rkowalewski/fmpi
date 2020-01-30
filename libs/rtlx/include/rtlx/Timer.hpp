#ifndef RTLX_TIMER_HPP
#define RTLX_TIMER_HPP

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

inline auto ChronoClockNow() -> double {
  using ClockType = ChooseClockType::type;
  using duration =
      std::chrono::duration<double, std::chrono::seconds::period>;
  return duration(ClockType::now().time_since_epoch()).count();
}

template <class Clock = ChooseClockType::type>
class Timer {
  using timepoint = typename Clock::time_point;
  using rep       = double;
  using duration  = std::chrono::duration<rep, std::chrono::seconds::period>;

 public:
  Timer(rep& marker)
    : _done(false)
    , _mark(marker)
    , _start(Clock::now()) {
  }

  ~Timer() {
    done();
  }

  void done() {
    if (_done) return;

    _stop = Clock::now();

    auto const d = std::chrono::duration_cast<duration>(_stop - _start);
    _mark += d.count();
    _done = true;
  }

 private:
  bool      _done;
  rep&      _mark;
  timepoint _start;
  timepoint _stop;
};
}  // namespace rtlx

#endif
