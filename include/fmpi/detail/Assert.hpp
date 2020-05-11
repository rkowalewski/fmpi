#ifndef FMPI_DETAIL_ASSERT_HPP_INCLUDED
#define FMPI_DETAIL_ASSERT_HPP_INCLUDED

#include <cstdlib>

#include "../Config.hpp"

namespace fmpi {
namespace detail {
// handles a failed assertion
void handle_failed_assert(
    const char* msg, const char* file, int line, const char* fnc) noexcept;

void handle_warning(
    const char* msg, const char* file, int line, const char* fnc) noexcept;

// note: debug assertion macros don't use fully qualified name
// because they should only be used in this library, where the whole namespace
// is available can be override via command line definitions
#if FMPI_DEBUG_ASSERT && !defined(FMPI_ASSERT)
#define FMPI_ASSERT(Expr)                                                   \
  static_cast<void>(                                                        \
      (Expr) ||                                                             \
      (detail::handle_failed_assert(                                        \
           "Assertion \"" #Expr "\" failed", __FILE__, __LINE__, __func__), \
       true))

#define FMPI_ASSERT_MSG(Expr, Msg)                           \
  static_cast<void>(                                         \
      (Expr) || (detail::handle_failed_assert(               \
                     "Assertion \"" #Expr "\" failed: " Msg, \
                     __FILE__,                               \
                     __LINE__,                               \
                     __func__),                              \
                 true))

#define FMPI_UNREACHABLE(Msg)   \
  detail::handle_failed_assert( \
      "Unreachable code reached: " Msg, __FILE__, __LINE__, __func__)

#define FMPI_WARNING(Msg) \
  detail::handle_warning(Msg, __FILE__, __LINE__, __func__)

#elif !defined(FMPI_ASSERT)
#define FMPI_ASSERT(Expr)
#define FMPI_ASSERT_MSG(Expr, Msg)
#define FMPI_UNREACHABLE(Msg) std::abort()
#define FMPI_WARNING(Msg)
#endif
}  // namespace detail
}  // namespace fmpi

#endif  // FMPI_DETAIL_ASSERT_HPP_INCLUDED
