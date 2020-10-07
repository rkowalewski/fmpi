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

void TraceStore::erase(std::string_view ctx) {
  // Test
  m_traces.erase(std::string(ctx));
}

MultiTrace::MultiTrace()
  : MultiTrace(anonymous) {
}

MultiTrace::MultiTrace(std::string_view ctx)
  : name_(ctx) {
}

MultiTrace::duration_t& MultiTrace::duration(std::string_view key) {
  return value<MultiTrace::duration_t>(key);
}

std::string_view MultiTrace::name() const noexcept {
  return name_;
}

#if 0
void MultiTrace::merge(MultiTrace&& source) {
  for (auto&& s : source.values_) {
    FMPI_ASSERT(values_.find(s.first) == std::end(values_));
  }
  // TODO: use flat_map::merge(T&&) instead of copy
  values_.insert(std::begin(source.values_), std::end(source.values_));
  source.values_.clear();
}
#endif

MultiTrace::~MultiTrace() {
  FMPI_DBG_STREAM("destroying multitrace: " << name_);
  if constexpr (kEnableTrace) {
    if (!name_.empty() && name_ != anonymous) {
      auto& global_instance = TraceStore::instance();
      // FMPI_ASSERT(!global_instance.contains(name_));
      global_instance.insert(name_, std::begin(values_), std::end(values_));
    }
  }
}

}  // namespace fmpi
