#ifndef FMPI_DEBUG_HPP
#define FMPI_DEBUG_HPP

#if FMPI_DEBUG_ASSERT

#include <dbg.h>
#include <mpi.h>

#include <fmpi/Config.hpp>

namespace fmpi {
namespace detail {

class DebugOutput {
 public:
  DebugOutput(
      const char* filepath,
      int         line,
      const char* function_name,
      const char* expression)
    : m_use_colorized_output(dbg_macro::isColorizedOutputEnabled())
    , m_filepath(filepath)
    , m_line(line)
    , m_function_name(function_name)
    , m_expression(expression) {
    const std::size_t path_length = m_filepath.length();
    if (path_length > MAX_PATH_LENGTH) {
      m_filepath = ".." + m_filepath.substr(
                              path_length - MAX_PATH_LENGTH, MAX_PATH_LENGTH);
    }
  }

  template <typename T>
  auto print(const std::string& type, T&& value) const -> T&&  // NOLINT
  {
    int flag = 0;

    int me = MPI_PROC_NULL;
    MPI_Initialized(&flag);

    const T&          ref = value;
    std::stringstream stream_value;
    const bool        print_expr_and_type =
        ::dbg_macro::pretty_print(stream_value, ref);

    std::stringstream output;
    output << ansi(ANSI_DEBUG) << "[" << m_filepath << ":" << m_line << " ("
           << m_function_name << ")] " << ansi(ANSI_RESET);

    if (flag) {
      MPI_Comm_rank(MPI_COMM_WORLD, &me);

      output << ansi(ANSI_DEBUG) << "[rank " << me << "] "
             << ansi(ANSI_RESET);
    }
    if (print_expr_and_type) {
      output << ansi(ANSI_EXPRESSION) << m_expression << ansi(ANSI_RESET)
             << " = ";
    }
    output << ansi(ANSI_VALUE) << stream_value.str() << ansi(ANSI_RESET);
    if (print_expr_and_type) {
      output << " (" << ansi(ANSI_TYPE) << type << ansi(ANSI_RESET) << ")";
    }

    output << std::endl;
    std::cerr << output.str();

    return std::forward<T>(value);
  }

 private:
  auto ansi(const char* code) const -> const char* {
    if (m_use_colorized_output) {
      return code;
    }

    return ANSI_EMPTY;
  }

  const bool m_use_colorized_output;

  std::string       m_filepath;
  const int         m_line;
  const std::string m_function_name;
  const std::string m_expression;

  static constexpr std::size_t MAX_PATH_LENGTH = 30;

  static constexpr const char* const ANSI_EMPTY      = "";
  static constexpr const char* const ANSI_DEBUG      = "\x1b[02m";
  static constexpr const char* const ANSI_EXPRESSION = "\x1b[36m";
  static constexpr const char* const ANSI_VALUE      = "\x1b[01m";
  static constexpr const char* const ANSI_TYPE       = "\x1b[32m";
  static constexpr const char* const ANSI_RESET      = "\x1b[0m";
};

}  // namespace detail
}  // namespace fmpi

namespace std {

template <class F, class T>
auto operator<<(std::ostream& os, std::pair<F, T> const& p) -> std::ostream& {
  os << std::string("(") << p.first << ", " << p.second << ")";
  return os;
}
}  // namespace std
#endif

#if FMPI_DEBUG_ASSERT

#define FMPI_DBG(...)                                                   \
  fmpi::detail::DebugOutput(__FILE__, __LINE__, __func__, #__VA_ARGS__) \
      .print(dbg_macro::type_name<decltype(__VA_ARGS__)>(), (__VA_ARGS__))

#define FMPI_DBG_STREAM(expr) \
  do {                        \
    std::ostringstream os;    \
    os << expr; /* NOLINT */  \
    auto msg = os.str();      \
    FMPI_DBG(msg);            \
  } while (0)

#define FMPI_DBG_RANGE(f, l)                                                \
  do {                                                                      \
    std::ostringstream os;                                                  \
    using value_t__ =                                                       \
        typename std::iterator_traits<decltype(f)>::value_type;             \
    os << "[";                                                              \
    std::copy(f, std::prev(l), std::ostream_iterator<value_t__>(os, ", ")); \
    os << *std::prev(l);                                                    \
    os << "]";                                                              \
    auto range = os.str();                                                  \
    FMPI_DBG(range);                                                        \
  } while (0)

#else

#define FMPI_DBG(...)               \
  do {                              \
    static_cast<void>(__VA_ARGS__); \
  } while (0)

#define FMPI_DBG_RANGE(f, l) \
  do {                       \
    static_cast<void>(f);    \
    static_cast<void>(l);    \
  } while (0)

#define FMPI_DBG_STREAM(...)
#endif

#if 0
#if defined(NDEBUG)
#define FMPI_ASSERT(expr) \
  (false ? static_cast<void>(expr) : static_cast<void>(0))
#else
#include <exception>
#define FMPI_ASSERT(expr)                                  \
  (RTLX_PREDICT_TRUE((expr)) ? static_cast<void>(0) : [] { \
    int me;                                                \
    MPI_Comm_rank(MPI_COMM_WORLD, &me);                    \
    std::ostringstream os;                                 \
    os << "[rank " << me << "] " << #expr;                 \
    throw std::runtime_error{os.str()};                    \
  }())  // NOLINT
#endif
#endif

#endif
