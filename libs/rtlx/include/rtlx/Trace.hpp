#ifndef RTLX_TRACE_HPP
#define RTLX_TRACE_HPP

#include <iosfwd>
#include <memory>
#include <mutex>
#include <rtlx/Timer.hpp>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rtlx {

class Trace;

class TraceStore {
 public:
  using duration = std::chrono::nanoseconds;
  using key_t    = std::string;
  using value_t  = std::variant<duration, int>;

 private:
  using context_t = key_t;
  using store_t   = std::unordered_map<key_t, value_t>;

  friend class Trace;

 public:
  TraceStore()                      = default;
  TraceStore(const TraceStore& src) = delete;
  auto operator=(const TraceStore& rhs) -> TraceStore& = delete;

  auto traces(context_t const& ctx)
      -> std::unordered_map<key_t, value_t> const&;
  // void clear();
  static constexpr auto enabled() noexcept -> bool {
#ifdef RTLX_ENABLE_TRACE
    return true;
#else
    return false;
#endif
  }
  void erase(context_t const& ctx);

  // Singleton Instance (Thread Safe)
  static auto instance() -> TraceStore&;

 private:
  std::unordered_map<context_t, std::unordered_map<key_t, value_t>> m_traces;
};

class Trace {
 public:
  using key_t   = TraceStore::key_t;
  using value_t = TraceStore::value_t;

 public:
  explicit Trace(TraceStore::context_t ctx);

  Trace(Trace&&)      = delete;
  Trace(Trace const&) = delete;
  Trace& operator=(Trace const&) = delete;
  Trace& operator=(Trace&&) = delete;

  ~Trace();

  static auto enabled() noexcept -> bool;

  void put(std::string_view, int v);

  template <class Rep, class Period>
  void add_time(
      std::string_view key, const std::chrono::duration<Rep, Period>& d) {
    using duration = typename TraceStore::duration;

    if constexpr (TraceStore::enabled()) {  // NOLINT
      auto& val = std::get<duration>(m_cache[key_t{key}]);
      val += duration{d};
    }
  }

  auto measurements() const -> std::unordered_map<key_t, value_t> const&;

  auto context() const noexcept -> std::string const&;

  void clear();

 private:
  std::string const                  m_context{};
  std::unordered_map<key_t, value_t> m_cache{};
};

template <class Clock = ChooseClockType::type>
class TimeTrace {
  using timer = Timer<Clock>;

 public:
  TimeTrace(Trace& trace, std::string_view key)
    : duration_(0)
    , timer_(duration_)
    , cache_(trace)
    , key_(std::string{key}) {
  }

  TimeTrace(TimeTrace&&)      = delete;
  TimeTrace(TimeTrace const&) = delete;
  TimeTrace& operator=(TimeTrace const&) = delete;
  TimeTrace& operator=(TimeTrace&&) = delete;

  ~TimeTrace() {
    finish();
  }

  void finish() noexcept {
    timer_.finish();
    cache_.add_time(key_, duration_);
  }

 private:
  typename Clock::duration duration_{};
  timer                    timer_;
  Trace&                   cache_;
  std::string              key_;
};

}  // namespace rtlx

#endif
