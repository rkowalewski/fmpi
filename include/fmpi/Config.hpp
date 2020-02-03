#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <mpi.h>

#include <cstddef>

namespace fmpi {

constexpr const char TOTAL[]         = "Ttotal";
constexpr const char COMPUTATION[]         = "Tcomp";
constexpr const char COMMUNICATION[] = "Tcomm";

constexpr int EXCH_TAG_RING  = 110435;
constexpr int EXCH_TAG_BRUCK = 110436;

// Cache Configuration
constexpr std::size_t kCacheLineSize      = 64;
constexpr std::size_t kCacheLineAlignment = 64;

// MPI Thread Level
constexpr auto kMpiThreadLevel = MPI_THREAD_SERIALIZED;

constexpr std::size_t kContainerStackSize = 1024 * 1024;

void initialize(int*, char*** argv);
void finalize();

// Only allowed as singleton object
struct Config {
  int main_core{};
  int dispatcher_core{};
  int scheduler_core{};
  int comp_core{};

  static Config const& instance();

 private:
  Config();

 public:
  Config(Config const&) = delete;
  Config& operator=(Config const&) = delete;

  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
};

std::ostream& operator<<(std::ostream&, const Config&);

void print_config(std::ostream& os);

}  // namespace fmpi
#endif
