#ifndef FMPI_CONCURRENCY_ASYNC_HPP
#define FMPI_CONCURRENCY_ASYNC_HPP

#include <future>

//#include <fmpi/Pinning.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/Function.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Timer.hpp>

namespace fmpi {

namespace detail {

template <typename R, typename F, typename... Args>
constexpr void set_promise_value(std::promise<R>& p, F&& f, Args&&... args) {
  if constexpr (std::is_void_v<R>) {
    {
      std::forward<F>(f)(std::forward<Args...>(args)...);
      p.set_value();
    }
  } else {
    auto value = std::forward<F>(f)(std::forward<Args...>(args)...);
    p.set_value(std::move(value));
  }
}
}  // namespace detail

template <typename F, typename... Ts>
inline auto async(int /*core*/, F&& f, Ts&&... params)
    -> std::future<invoke_result_t<F, Ts...>> {
  using future_inner_type = invoke_result_t<F, Ts...>;

  auto promise_ptr = std::make_unique<std::promise<future_inner_type>>();
  auto result      = promise_ptr->get_future();
  auto thread      = std::thread([promise_ptr = std::move(promise_ptr),
                             f           = std::forward<F>(f),
                             params...]() mutable {
    try {
      detail::set_promise_value(
          *promise_ptr, std::forward<F>(f), std::forward<Ts...>(params)...);
    } catch (...) {
      promise_ptr->set_exception(std::current_exception());
    }
  });

  thread.detach();

  return result;
}
}  // namespace fmpi

#endif
