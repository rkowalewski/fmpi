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

template <class Clock = ChooseClockType::type>
class Timer {
  using timepoint = typename Clock::time_point;

 public:
  using duration = typename Clock::duration;

  constexpr Timer(duration& marker)
    : _done(false)
    , _mark(marker)
    , _start(Clock::now()) {
  }

  ~Timer() {
    finish();
  }

  void finish() {
    if (_done) return;

    _stop = Clock::now();

    _mark += _stop - _start;
    _done = true;
  }
  bool done() const noexcept {
    return _done;
  }

 private:
  bool      _done;
  duration& _mark;
  timepoint _start;
  timepoint _stop;
};

template <class Rep, class Period>
constexpr double to_seconds(const std::chrono::duration<Rep, Period>& d) {
  using duration =
      std::chrono::duration<double, std::chrono::seconds::period>;

  return std::chrono::duration_cast<duration>(d).count();
}

}  // namespace rtlx

#endif
