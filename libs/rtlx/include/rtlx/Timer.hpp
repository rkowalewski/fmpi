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

  static_assert(
      std::is_same<typename Clock::duration, std::chrono::nanoseconds>::value,
      "inprecise clock");

 public:
  using duration = typename Clock::duration;

  constexpr explicit Timer(duration& marker)
    : _done(false)
    , _mark(marker)
    , _start(Clock::now()) {
  }

  Timer(Timer&&)      = delete;
  Timer(Timer const&) = delete;
  Timer& operator=(Timer const&) = delete;
  Timer& operator=(Timer&&) = delete;

  ~Timer() {
    finish();
  }

  void finish() {
    if (_done) {
      return;
    }

    _stop = Clock::now();

    _mark += std::chrono::duration_cast<duration>(_stop - _start);

    _done = true;
  }
  [[nodiscard]] bool done() const noexcept {
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

template <class Clock, class F>
auto timed_exection() -> typename Clock::duration {
}

}  // namespace rtlx

#endif
