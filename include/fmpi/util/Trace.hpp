#ifndef FMPI_UTIL_TRACE_HPP
#define FMPI_UTIL_TRACE_HPP
#include <boost/container/flat_map.hpp>
#include <chrono>
#include <fmpi/Config.hpp>
#include <fmpi/detail/Assert.hpp>
#include <rtlx/Timer.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace fmpi {

class MultiTrace;

class TraceStore {
 public:
  /// values
  using duration_t = std::chrono::nanoseconds;
  using integer_t  = int64_t;

  using key_type = std::string;
  // using mapped_type = std::variant<duration_t, integer_t>;
  using mapped_type = duration_t;

 private:
  using multi_trace = std::unordered_map<key_type, mapped_type>;

  using context   = std::string;
  using container = std::unordered_map<context, multi_trace>;

  TraceStore();

 public:
  TraceStore(const TraceStore& src) = delete;
  TraceStore& operator=(const TraceStore& rhs) = delete;

  void insert(std::string_view ctx, multi_trace const& values);
  void merge(std::string_view ctx, multi_trace const& values);

  void erase(std::string_view /*ctx*/);

  [[nodiscard]] bool empty() const noexcept {
    return m_traces.empty();
  }

  [[nodiscard]] bool contains(std::string_view key) const {
    auto it = m_traces.find(context(key));
    return it != std::end(m_traces) && !it->second.empty();
  }

  multi_trace const& traces(std::string_view ctx);

  // Singleton Instance (Thread Safe)
  static auto instance() -> TraceStore&;

 private:
  container m_traces;
};

class MultiTrace {
  using duration_t = typename TraceStore::duration_t;
  using integer_t  = typename TraceStore::integer_t;

  using mapped_type = TraceStore::mapped_type;

  static constexpr auto anonymous = std::string_view("<anonymous>");

 public:
  using cache = std::unordered_map<std::string, mapped_type>;

  explicit MultiTrace();
  explicit MultiTrace(std::string_view ctx);

  MultiTrace(const MultiTrace& other) = delete;
  MultiTrace(MultiTrace&& other)      = delete;
  MultiTrace& operator=(const MultiTrace& other) = delete;
  MultiTrace& operator=(MultiTrace&& other) = delete;

  ~MultiTrace();

  duration_t& duration(std::string_view key);

  cache const& values() const noexcept {
    return values_;
  }

 private:
  cache values_{};
  // Yes we can use a string view here because
  std::string const name_;
};

using steady_timer = rtlx::steady_timer;

}  // namespace fmpi
#endif
