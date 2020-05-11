#ifndef RTLX_TIMER_HPP
#define RTLX_TIMER_HPP

#include <chrono>

#include <rtlx/Assert.hpp>

namespace rtlx {

struct ChooseClockType {
  using type = std::conditional_t<
      std::chrono::high_resolution_clock::is_steady,
      std::chrono::high_resolution_clock,
      std::chrono::steady_clock>;
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
    if (_state != state::running) return;

    _mark += Clock::now() - _start;
    _state = state::paused;
  }

  void resume() noexcept {
    if (_state != state::paused) return;

    _start = Clock::now();
    _state = state::running;
  }

  void finish() {
    if (_state == state::stopped) {
      return;
    } else if (_state == state::running) {
      _mark += Clock::now() - _start;
    }

    _state = state::stopped;
  }
  [[nodiscard]] bool done() const noexcept {
    return _state == state::stopped;
  }

  [[nodiscard]] bool running() const noexcept {
    return _state == state::running;
  }
};

template <class Timer>
class ScopedTimerSwitch {
  Timer& current_;
  Timer& other_;

 public:
  ScopedTimerSwitch(Timer& current, Timer& other) noexcept
    : current_(current)
    , other_(other) {
    current_.pause();
    other_.resume();
  }
  ~ScopedTimerSwitch() {
    other_.pause();
    current_.resume();
  }
};

using steady_timer = Timer<>;

template <class Rep, class Period>
constexpr double to_seconds(const std::chrono::duration<Rep, Period>& d) {
  using duration =
      std::chrono::duration<double, std::chrono::seconds::period>;

  return std::chrono::duration_cast<duration>(d).count();
}

}  // namespace rtlx

#endif
