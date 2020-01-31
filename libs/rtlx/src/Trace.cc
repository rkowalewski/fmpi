#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <rtlx/Assert.hpp>
#include <rtlx/Timer.hpp>
#include <rtlx/Trace.hpp>

namespace rtlx {

Trace::Trace(TraceStore::context_t ctx)
  : m_context(std::move(ctx)) {
}

Trace::~Trace() {
  if constexpr (TraceStore::enabled()) {
    auto &store = TraceStore::GetInstance();
    store.m_traces[m_context].insert(std::begin(m_cache), std::end(m_cache));
  }
}

auto Trace::context() const noexcept -> std::string const & {
  return m_context;
}

void Trace::put(TraceStore::key_t const &key, int v) {
  if constexpr (TraceStore::enabled()) {
    m_cache[key] = v;
  }
}

auto TraceStore::traces(context_t const &ctx)
    -> std::unordered_map<TraceStore::key_t, TraceStore::value_t> const & {
  return m_traces[ctx];
}

auto TraceStore::GetInstance() -> TraceStore & {
  std::call_once(
      m_onceFlag, [] { m_instance = std::make_unique<TraceStore>(); });
  return *m_instance;
}

void TraceStore::erase(context_t const &ctx) {
  // Test
  m_traces.erase(ctx);
}

/// Static variables

std::unique_ptr<TraceStore> TraceStore::m_instance{};

std::once_flag TraceStore::m_onceFlag{};
}  // namespace rtlx
