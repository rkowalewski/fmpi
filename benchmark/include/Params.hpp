#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <chrono>
#include <iosfwd>
#include <string>

namespace fmpi {
namespace benchmark {

struct Params {
 private:
  // 5 seconds is the limit for a single round
  static constexpr auto five_seconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::seconds(5));

 public:
  Params();
  uint32_t                  nhosts{};
  unsigned int              niters{};
  std::size_t               smin       = 1u << 5;  // 32 bytes
  std::size_t               smax       = 1u << 7;  // 128 bytes
  uint32_t                  pmin       = 1u << 1;
  uint32_t                  pmax       = pmin;
  std::chrono::microseconds time_limit = five_seconds;

  std::string pattern{};     // pattern for algorithm selection
  bool        check{false};  // validate correctness
};

bool read_input(int argc, char* argv[], struct Params& params);

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim = "\n");

}  // namespace benchmark
}  // namespace fmpi
#endif
