#include <fmpi/Debug.hpp>
#include <fmpi/util/Trace.hpp>

namespace fmpi {

TraceStore::TraceStore() = default;

TraceStore::multi_trace const& TraceStore::traces(std::string_view ctx) {
  return m_traces[std::string(ctx)];
}

auto TraceStore::instance() -> TraceStore& {
  static TraceStore singleton;
  return singleton;
}

void TraceStore::insert(std::string_view ctx, multi_trace const& values) {
  if constexpr (kEnableTrace) {
    m_traces[context(ctx)].insert(values.begin(), values.end());
  }
}

void TraceStore::erase(std::string_view ctx) {
  m_traces.erase(std::string(ctx));
}

MultiTrace::MultiTrace()
  : MultiTrace(anonymous) {
}

MultiTrace::MultiTrace(std::string_view ctx)
  : name_(ctx) {
}

MultiTrace::duration_t& MultiTrace::duration(std::string_view key) {
  return values_[std::string(key)];
}

MultiTrace::~MultiTrace() {
  FMPI_DBG_STREAM("destroying multitrace: " << name_);
  if constexpr (kEnableTrace) {
    if (!name_.empty() && name_ != anonymous) {
      auto& global_instance = TraceStore::instance();
      FMPI_ASSERT(!global_instance.contains(name_));
      global_instance.insert(name_, values_);
    }
  }
}

}  // namespace fmpi
