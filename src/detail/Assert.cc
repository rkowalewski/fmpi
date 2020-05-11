#include <fmpi/detail/Assert.hpp>

#include <cstdio>
#include <cstdlib>


using namespace fmpi;

void detail::handle_failed_assert(
    const char* msg, const char* file, int line, const char* fnc) noexcept {
  std::fprintf(
      stderr,
      "[%s] Assertion failure in function %s (%s:%d): %s.\n",
      FMPI_LOG_PREFIX,
      fnc,
      file,
      line,
      msg);
}

void detail::handle_warning(
    const char* msg, const char* file, int line, const char* fnc) noexcept {
  std::fprintf(
      stderr,
      "[%s] Warning triggered in function %s (%s:%d): %s.\n",
      FMPI_LOG_PREFIX,
      fnc,
      file,
      line,
      msg);
}
