#ifndef FMPI_CONFIG_HPP
#define FMPI_CONFIG_HPP

#include <mpi.h>

#include <cstddef>
#include <iosfwd>
#include <string_view>

#ifndef __cpp_lib_hardware_interference_size
namespace std {

//  mimic: std::hardware_constructive_interference_size, C++17
constexpr std::size_t hardware_constructive_interference_size = 64;

//  mimic: std::hardware_destructive_interference_size, C++17
constexpr std::size_t hardware_destructive_interference_size = 128;

}  // namespace std
#endif

namespace fmpi {

constexpr auto TOTAL         = std::string_view{"Ttotal"};
constexpr auto COMPUTATION   = std::string_view{"Tcomp"};
constexpr auto COMMUNICATION = std::string_view{"Tcomm"};
constexpr auto N_COMM_ROUNDS = std::string_view{"Ncomm_rounds"};

constexpr int EXCH_TAG_RING  = 110435;
constexpr int EXCH_TAG_BRUCK = 110436;

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

static_assert(std::hardware_destructive_interference_size >= max_align_v);

// MPI Thread Level
constexpr auto kMpiThreadLevel = MPI_THREAD_SERIALIZED;

constexpr std::size_t kContainerStackSize      = 1024 * 512;
constexpr std::size_t kMaxContiguousBufferSize = 1024 * 512;

constexpr std::size_t kCacheLineAlignment =
    std::hardware_destructive_interference_size;

constexpr std::size_t kCacheLineSize =
    std::hardware_constructive_interference_size;

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
