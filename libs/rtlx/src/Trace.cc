#include <rtlx/Assert.h>
#include <rtlx/Timer.h>
#include <rtlx/Trace.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

namespace rtlx {

auto TimeTrace::enabled() noexcept -> bool
{
  auto &store = TraceStore::GetInstance();
  return store.enabled();
}

TimeTrace::TimeTrace(TraceStore::context_t ctx)
  : m_context(std::move(ctx))
{
}

auto TimeTrace::context() const noexcept -> std::string const &
{
  return m_context;
}

void TimeTrace::tick(const TraceStore::key_t &key)
{
  if (enabled()) {
    RTLX_ASSERT(m_cache.find(key) == std::end(m_cache));
    m_cache[key] = ChronoClockNow();
  }
}

void TimeTrace::tock(const TraceStore::key_t &key)
{
  auto &store = TraceStore::GetInstance();
  if (store.enabled()) {
    RTLX_ASSERT(m_cache.find(key) != std::end(m_cache));
    auto &v = std::get<double>(store.traces(m_context)[key]);
    v += ChronoClockNow() - std::get<double>(m_cache[key]);
    m_cache.erase(key);
  }
}

void TimeTrace::put(TraceStore::key_t const &key, int v) const
{
  auto &store = TraceStore::GetInstance();
  if (store.enabled()) {
    store.traces(m_context)[key] = v;
  }
}

auto TraceStore::traces(context_t const &ctx)
    -> std::unordered_map<TraceStore::key_t, TraceStore::value_t> &
{
  return m_traces[ctx];
}

static auto isTraceEnvironFlagEnabled() -> bool
{
  // Split into key and value:
  if (auto const *flag = std::getenv("RTLX_ENABLE_TRACE")) {
    std::string flag_value = flag;
    return flag_value == "1" || flag_value == "ON";
  }

  return false;
}

auto TraceStore::GetInstance() -> TraceStore &
{
  std::call_once(m_onceFlag, [] {
    m_instance              = std::make_unique<TraceStore>();
    (*m_instance).m_enabled = isTraceEnvironFlagEnabled();
  });
  return *m_instance;
}

void TraceStore::clear()
{
  m_traces.clear();
}

void TraceStore::erase(context_t const &ctx)
{
  if (auto it = m_traces.find(ctx); it != m_traces.end()) {
    m_traces.erase(it);
  }
}

auto TraceStore::enabled() const noexcept -> bool
{
  return m_enabled;
}

/// Static variables

std::unique_ptr<TraceStore> TraceStore::m_instance{};

std::once_flag TraceStore::m_onceFlag{};
}  // namespace rtlx
