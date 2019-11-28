#ifndef RTLX_TRACE_H
#define RTLX_TRACE_H

#include <rtlx/Timer.h>

#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rtlx {

class TimeTrace;

class TraceStore {
  using key_t     = std::string;
  using context_t = key_t;
  using value_t   = std::variant<double, int>;
  using store_t   = std::unordered_map<key_t, value_t>;

  friend class TimeTrace;

 public:
  TraceStore()                      = default;
  TraceStore(const TraceStore &src) = delete;
  auto operator=(const TraceStore &rhs) -> TraceStore & = delete;

  auto traces(context_t const &ctx) -> std::unordered_map<key_t, value_t> const &;
  //void clear();
  static constexpr auto enabled() noexcept -> bool {
#if defined(RTLX_ENABLE_TRACE) && (RTLX_ENABLE_TRACE == 1)
    return true;
#else
    return false;
#endif
  }
  void erase(context_t const &ctx);

  // Singleton Instance (Thread Safe)
  static auto GetInstance() -> TraceStore &;

 private:
  static std::unique_ptr<TraceStore> m_instance;
  static std::once_flag              m_onceFlag;

  std::unordered_map<context_t, std::unordered_map<key_t, value_t>> m_traces;
};

class TimeTrace {
 public:
  using key_t   = TraceStore::key_t;
  using value_t = TraceStore::value_t;

 public:
  TimeTrace(TraceStore::context_t ctx);

  static auto enabled() noexcept -> bool;

  void tick(const key_t &key);
  void tock(const key_t &key);

  void put(key_t const & /*key*/, int v) const;

  auto measurements() const -> std::unordered_map<key_t, value_t> const &;

  auto context() const noexcept -> std::string const &;

  void clear();

 private:
  std::string const                  m_context{};
  std::unordered_map<key_t, value_t> m_cache{};
};

}  // namespace rtlx

#endif
