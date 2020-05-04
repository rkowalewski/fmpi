#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <rtlx/Assert.hpp>
#include <rtlx/Timer.hpp>
#include <rtlx/Trace.hpp>

namespace rtlx {

Trace::Trace(std::string_view ctx)
  : m_context(ctx) {
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

void Trace::put(std::string_view key, TraceStore::integer_t v) {
  if constexpr (TraceStore::enabled()) {
    m_cache[key_t{key}] = v;
  }
}

auto TraceStore::traces(std::string_view ctx) noexcept
    -> std::unordered_map<TraceStore::key_t, TraceStore::value_t> const& {
  return m_traces[std::string(ctx)];
}

auto TraceStore::traces() const noexcept -> std::unordered_map<
    TraceStore::context_t,
    std::unordered_map<TraceStore::key_t, TraceStore::value_t>> const& {
  return m_traces;
}

auto TraceStore::instance() -> TraceStore& {
  static TraceStore singleton;
  return singleton;
}

void TraceStore::erase(std::string_view ctx) {
  // Test
  m_traces.erase(std::string(ctx));
}

}  // namespace rtlx
