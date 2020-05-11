#ifndef FMPI_UTIL_TRACE_HPP
#define FMPI_UTIL_TRACE_HPP
#include <chrono>
#include <string>
#include <tuple>
#include <variant>

#include <boost/container/flat_map.hpp>

#include <rtlx/Timer.hpp>

#include <fmpi/Config.hpp>

namespace fmpi {

class TraceStore {
 public:
  /// values
  using duration_t = std::chrono::nanoseconds;
  using integer_t  = int64_t;

  using key_type    = std::string_view;
  using mapped_type = std::variant<duration_t, integer_t>;

 private:
  using multi_trace = boost::container::flat_map<key_type, mapped_type>;

  using context   = std::string;
  using container = boost::container::flat_map<context, multi_trace>;

  TraceStore();

 public:
  TraceStore(const TraceStore& src) = delete;
  TraceStore& operator=(const TraceStore& rhs) = delete;

  template <class InputIterator>
  void insert(std::string_view ctx, InputIterator first, InputIterator last) {
    if constexpr (kEnableTrace) {
      if (std::distance(first, last) > 0) {
        m_traces[context(ctx)].insert(first, last);
      }
    }
  }

  void erase(std::string_view);

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

  using cache = boost::container::flat_map<TraceStore::key_type, mapped_type>;

 public:
  explicit MultiTrace(std::string_view ctx);

  MultiTrace(const MultiTrace& other) = delete;
  MultiTrace(MultiTrace&& other)      = delete;
  MultiTrace& operator=(const MultiTrace& other) = delete;
  MultiTrace& operator=(MultiTrace&& other) = delete;

  ~MultiTrace();

  template <class T>
  T& value(std::string_view key) {
    auto it = values_.find(key);

    if (it == std::end(values_)) {
      std::tie(it, std::ignore) = values_.insert_or_assign(key, T{});
    }

    return std::get<T>(it->second);
  }

  duration_t& duration(std::string_view key);

  [[nodiscard]] std::string_view name() const noexcept;

  void merge(MultiTrace&& source);

 private:
  cache            values_{};
  std::string_view name_;
};

using steady_timer = rtlx::steady_timer;

}  // namespace fmpi
#endif
