#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <mpi.h>

#include <cstddef>
#include <iosfwd>

namespace fmpi {

constexpr const char TOTAL[]         = "Ttotal";
constexpr const char COMPUTATION[]   = "Tcomp";
constexpr const char COMMUNICATION[] = "Tcomm";
constexpr const char N_COMM_ROUNDS[] = "Ncomm_rounds";

constexpr int EXCH_TAG_RING  = 110435;
constexpr int EXCH_TAG_BRUCK = 110436;

// MPI Thread Level
constexpr auto kMpiThreadLevel = MPI_THREAD_SERIALIZED;

constexpr std::size_t kContainerStackSize = 1024 * 512;
constexpr std::size_t kMaxContiguousBufferSize = 1024 * 512;

void initialize(int*, char*** argv);
void finalize();

// Only allowed as singleton object
struct Config {
  int main_core{};
  int dispatcher_core{};
  int scheduler_core{};
  int comp_core{};
  int domain_size{};
  int num_threads{};

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

namespace detail {

// Implemented this way because of a bug in Clang for ARMv7, which gives the
// wrong result for `alignof` a `union` with a field of each scalar type.
constexpr size_t max_align_(std::size_t a) {
  return a;
}
template <typename... Es>
constexpr std::size_t max_align_(std::size_t a, std::size_t e, Es... es) {
  return !(a < e) ? max_align_(a, es...) : max_align_(e, es...);
}
template <typename... Ts>
struct max_align_t_ {
  static constexpr std::size_t value = max_align_(0U, alignof(Ts)...);
};
using max_align_v_ = max_align_t_<
    long double,
    double,
    float,
    long long int,
    long int,
    int,
    short int,
    bool,
    char,
    char16_t,
    char32_t,
    wchar_t,
    void*,
    std::max_align_t>;

}  // namespace detail

constexpr std::size_t max_align_v = detail::max_align_v_::value;
struct alignas(max_align_v) max_align_t {};

//  mimic: std::hardware_constructive_interference_size, C++17
constexpr std::size_t hardware_constructive_interference_size = 64;

//  mimic: std::hardware_destructive_interference_size, C++17
constexpr std::size_t hardware_destructive_interference_size = 128;

static_assert(hardware_destructive_interference_size >= max_align_v);

constexpr std::size_t kCacheLineAlignment =
    hardware_destructive_interference_size;
constexpr std::size_t kCacheLineSize =
    hardware_constructive_interference_size;

}  // namespace fmpi
#endif
