#ifndef RTLX_SCOPEDLAMBDA_HPP
#define RTLX_SCOPEDLAMBDA_HPP
#include <type_traits>
#include <utility>

namespace rtlx {
namespace detail {
template <typename LambdaT>
class lambda_call {
 public:
  lambda_call(const lambda_call&) = delete;
  lambda_call& operator=(const lambda_call&) = delete;
  lambda_call& operator=(lambda_call&& other) = delete;

  explicit lambda_call(LambdaT&& lambda) noexcept
    : m_lambda(std::move(lambda)) {
    static_assert(
        std::is_same<decltype(lambda()), void>::value,
        "scope_exit lambdas must not have a return value");
    static_assert(
        !std::is_lvalue_reference<LambdaT>::value &&
            !std::is_rvalue_reference<LambdaT>::value,
        "scope_exit should only be directly used with a lambda");
  }

  lambda_call(lambda_call&& other) noexcept
    : m_lambda(std::move(other.m_lambda))
    , m_call(other.m_call) {
    other.m_call = false;
  }

  ~lambda_call() noexcept {
    reset();
  }

  // Ensures the scope_exit lambda will not be called
  void release() noexcept {
    m_call = false;
  }

  // Executes the scope_exit lambda immediately if not yet run; ensures it
  // will not run again
  void reset() noexcept {
    if (m_call) {
      m_call = false;
      m_lambda();
    }
  }

  // Returns true if the scope_exit lambda is still going to be executed
  explicit operator bool() const noexcept {
    return m_call;
  }

 private:
  LambdaT m_lambda;
  bool    m_call = true;
};
}  // namespace detail

template <typename LambdaT>
inline auto scope_exit(LambdaT&& lambda) noexcept {
  return detail::lambda_call<LambdaT>(std::forward<LambdaT>(lambda));
}
}  // namespace rtlx

#endif
