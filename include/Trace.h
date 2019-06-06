#ifndef TRACE_H__INCLUDED
#define TRACE_H__INCLUDED

#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <Timer.h>

class TimeTrace;

class TraceStore {
  using key_t     = std::string;
  using context_t = key_t;
  using value_t   = double;

  friend class TimeTrace;

  TraceStore()                      = default;
  TraceStore(const TraceStore &src) = delete;
  TraceStore &operator=(const TraceStore &rhs) = delete;

 public:
  static TraceStore &                 GetInstance();
  std::unordered_map<key_t, value_t> &get(context_t const &ctx);
  void                                clear();
  bool                                enabled() const noexcept;

 private:
  static std::unique_ptr<TraceStore> m_instance;
  static std::once_flag              m_onceFlag;

  std::unordered_map<context_t, std::unordered_map<key_t, value_t>> m_traces;
  bool m_enabled{false};
};

class TimeTrace {
 public:
  TimeTrace(int pid, TraceStore::context_t ctx);

  bool enabled() const noexcept;

  void tick(TraceStore::key_t key);
  void tock(TraceStore::key_t key);

  std::unordered_map<TraceStore::key_t, TraceStore::value_t> const &measurements()
      const;

  int                pid() const noexcept;
  std::string const &context() const noexcept;

  void clear();

 private:
  int                                                        m_pid{};
  std::string const                                          m_context{};
  std::unordered_map<TraceStore::key_t, TraceStore::value_t> m_cache;
};

#endif
