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
    auto& store = TraceStore::instance();
    store.m_traces[m_context].insert(std::begin(m_cache), std::end(m_cache));
  }
}

auto Trace::context() const noexcept -> std::string const& {
  return m_context;
}

void Trace::put(std::string_view key, int v) {
  if constexpr (TraceStore::enabled()) {
    m_cache[key_t{key}] = v;
  }
}

auto TraceStore::traces(context_t const& ctx)
    -> std::unordered_map<TraceStore::key_t, TraceStore::value_t> const& {
  return m_traces[ctx];
}

auto TraceStore::instance() -> TraceStore& {
  static TraceStore singleton;
  return singleton;
}

void TraceStore::erase(context_t const& ctx) {
  // Test
  m_traces.erase(ctx);
}

}  // namespace rtlx
