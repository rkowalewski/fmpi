#ifndef RTLX_TRACE_H
#define RTLX_TRACE_H

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

  auto get(context_t const &ctx) -> std::unordered_map<key_t, value_t> &;
  void clear();
  auto enabled() const noexcept -> bool;

  // Singleton Instance (Thread Safe)
  static auto GetInstance() -> TraceStore &;

 private:
  static std::unique_ptr<TraceStore> m_instance;
  static std::once_flag              m_onceFlag;

  std::unordered_map<context_t, std::unordered_map<key_t, value_t>> m_traces;
  bool m_enabled{false};
};

class TimeTrace {
 public:
  using key_t   = TraceStore::key_t;
  using value_t = TraceStore::value_t;

 public:
  TimeTrace(int pid, TraceStore::context_t ctx);

  static auto enabled() noexcept -> bool;

  void tick(const TraceStore::key_t &key);
  void tock(const TraceStore::key_t &key);

  void put(TraceStore::key_t const &, int v) const;

  auto lookup(TraceStore::key_t const &key) const -> value_t;

  auto measurements() const
      -> std::unordered_map<TraceStore::key_t, TraceStore::value_t> const &;

  auto pid() const noexcept -> int;
  auto context() const noexcept -> std::string const &;

  void clear();

 private:
  int                                                        m_pid{};
  std::string const                                          m_context{};
  std::unordered_map<TraceStore::key_t, TraceStore::value_t> m_cache;
};

}  // namespace rtlx

#endif
