#ifndef FMPI_COMMON_PORTING_HPP
#define FMPI_COMMON_PORTING_HPP

#include <cstddef>
#include <thread>

#ifndef __cpp_lib_hardware_interference_size
namespace std {

//  mimic: std::hardware_constructive_interference_size, C++17
constexpr std::size_t hardware_constructive_interference_size = 64;

//  mimic: std::hardware_destructive_interference_size, C++17
constexpr std::size_t hardware_destructive_interference_size = 128;

}  // namespace std

#endif

namespace fmpi {

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

int  get_num_threads();
bool pinThreadToCore(std::thread& thread, int core_id);

}  // namespace fmpi

#endif
