#ifndef FMPI_PINNING_HPP
#define FMPI_PINNING_HPP

#include <cstdint>
#include <iosfwd>

namespace fmpi {

// Only allowed as singleton object
struct Pinning {
  uint32_t main_core{};
  uint32_t dispatcher_core{};
  uint32_t num_nodes{};
  // uint32_t scheduler_core{};
  // uint32_t comp_core{};
  uint32_t domain_size{};
  uint32_t num_threads{};
  bool     smt_enabled{};

  static Pinning const& instance();

 private:
  Pinning();

 public:
  Pinning(Pinning const&) = delete;
  Pinning& operator=(Pinning const&) = delete;

  Pinning(Pinning&&) = delete;
  Pinning& operator=(Pinning&&) = delete;
};

std::ostream& operator<<(std::ostream& /*os*/, const Pinning& /*pinning*/);

void print_pinning(std::ostream& os);

}  // namespace fmpi
#endif
