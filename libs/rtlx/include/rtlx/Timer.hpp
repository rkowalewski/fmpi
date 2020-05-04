#ifndef RTLX_TIMER_HPP
#define RTLX_TIMER_HPP

#include <chrono>

#include <rtlx/Assert.hpp>

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

  enum class state : uint8_t
  {
    running,
    paused,
    stopped
  };

  typename Clock::duration& _mark;
  timepoint                 _start;
  timepoint                 _stop;
  state                     _state;

 public:
  using duration = typename Clock::duration;
  using clock    = Clock;

  constexpr explicit Timer(duration& marker) noexcept
    : _mark(marker)
    , _start(Clock::now())
    , _state(state::running) {
  }

  Timer(Timer&&)      = delete;
  Timer(Timer const&) = delete;
  Timer& operator=(Timer const&) = delete;
  Timer& operator=(Timer&&) = delete;

  ~Timer() {
    finish();
  }

  void pause() noexcept {
    RTLX_ASSERT(_state == state::running);

    _mark += elapsed();
    _state = state::paused;
  }

  void resume() noexcept {
    RTLX_ASSERT(_state == state::paused);

    _start = Clock::now();
    _state = state::running;
  }

  void finish() noexcept {
    if (_state == state::stopped) {
      return;
    }

    RTLX_ASSERT(_state == state::running);

    _stop = Clock::now();

    _mark += elapsed();

    _state = state::stopped;
  }
  [[nodiscard]] bool done() const noexcept {
    return _state == state::stopped;
  }

 private:
  duration elapsed() const noexcept {
    return std::chrono::duration_cast<duration>(Clock::now() - _start);
  }
};

template <class Timer>
class TimerPauseResume {
  Timer& timer_;

 public:
  TimerPauseResume(Timer& timer)
    : timer_(timer) {
    timer_.pause();
  }
  ~TimerPauseResume() {
    timer_.resume();
  }
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
