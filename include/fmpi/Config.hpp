#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <mpi.h>

#include <cstddef>
#include <iosfwd>

namespace fmpi {

void initialize(int*, char*** argv);
void finalize();

// Only allowed as singleton object
struct Config {
  uint32_t main_core{};
  uint32_t dispatcher_core{};
  // uint32_t scheduler_core{};
  // uint32_t comp_core{};
  uint32_t domain_size{};
  uint32_t num_threads{};

  static Config const& instance();

 private:
  Config();

 public:
  Config(Config const&) = delete;
  Config& operator=(Config const&) = delete;

  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
};

std::ostream& operator<<(std::ostream& /*os*/, const Config& /*pinning*/);

void print_config(std::ostream& os);

}  // namespace fmpi
#endif
