#include <cstring>
#include <iostream>
#include <mutex>

#include <Debug.h>
#include <timer/Timer.h>
#include <timer/Trace.h>

extern char **environ;

namespace a2a {

bool TimeTrace::enabled() const noexcept
{
  auto &store = TraceStore::GetInstance();
  return store.enabled();
}

TimeTrace::TimeTrace(int pid, TraceStore::context_t ctx)
  : m_pid(pid)
  , m_context(std::move(ctx))
{
}

int TimeTrace::pid() const noexcept
{
  return m_pid;
}

std::string const &TimeTrace::context() const noexcept
{
  return m_context;
}

void TimeTrace::tick(const TraceStore::key_t& key)
{
  if (enabled()) {
    A2A_ASSERT(m_cache.find(key) == std::end(m_cache));
    m_cache[key] = ChronoClockNow();
  }
}

void TimeTrace::tock(const TraceStore::key_t& key)
{
  auto &store = TraceStore::GetInstance();
  if (store.enabled()) {
    A2A_ASSERT(m_cache.find(key) != std::end(m_cache));
    store.get(m_context)[key] += ChronoClockNow() - m_cache[key];
    m_cache.erase(key);
  }
}

void TimeTrace::clear()
{
  auto &store = TraceStore::GetInstance();
  store.get(m_context).clear();
  m_cache.clear();
}

std::unordered_map<TraceStore::key_t, TraceStore::value_t> const &
TimeTrace::measurements() const
{
  auto &      store = TraceStore::GetInstance();
  auto const &m     = store.get(m_context);
  // A2A_ASSERT(m.size() > 0);
  return m;
}

TraceStore::value_t TimeTrace::lookup(TraceStore::key_t const &key) const
{
  if (!enabled()) {
    return TraceStore::value_t{};
  }

  auto const &m  = measurements();
  auto        it = m.find(key);

  if (it == m.end()) {
    return TraceStore::value_t{};
  }
  return it->second;
}

std::unordered_map<TraceStore::key_t, TraceStore::value_t> &TraceStore::get(
    context_t const &ctx)
{
  return m_traces[ctx];
}

static bool isTraceEnvironFlagEnabled()
{
  char *env_var_kv = *environ;

  int i = 1;
  for (; env_var_kv != nullptr; ++i) {
    // Split into key and value:
    char *      flag_name_cstr  = env_var_kv;
    char *      flag_value_cstr = std::strstr(env_var_kv, "=");
    auto         flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if ((std::strstr(flag_name.c_str(), "A2A_ENABLE_TRACE") != nullptr)) {
      return flag_value == "1" || flag_value == "ON";
    }

    env_var_kv = *(environ + i);
  }

  return false;
}

TraceStore &TraceStore::GetInstance()
{
  std::call_once(m_onceFlag, [] {
    m_instance.reset(new TraceStore);
    (*m_instance).m_enabled = isTraceEnvironFlagEnabled();
  });
  return *m_instance;
}

void TraceStore::clear()
{
  m_traces.clear();
}

bool TraceStore::enabled() const noexcept
{
  return m_enabled;
}

/// Static variables

std::unique_ptr<TraceStore> TraceStore::m_instance{};

std::once_flag TraceStore::m_onceFlag{};
}  // namespace a2a
