#ifndef FMPI_DETAIL_ASYNC_HPP
#define FMPI_DETAIL_ASYNC_HPP

#include <future>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/detail/Capture.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Timer.hpp>

namespace fmpi {

namespace detail {

template <typename RET>
constexpr void call_with_promise(
    fmpi::Function<RET()>&& callable, std::promise<RET>& pr) {
  if constexpr (std::is_void_v<RET>) {
    callable();
    pr.set_value();
  } else {
    pr.set_value(callable());
  }
}

}  // namespace detail

template <typename R, typename F, typename... Ts>
inline std::future<R> async(int core, F&& f, Ts&&... params) {
  auto lambda =
      fmpi::makeCapture<R>(std::forward<F>(f), std::forward<Ts>(params)...);

  using function_t = fmpi::Function<R()>;

  auto task = function_t{std::move(lambda)};
  auto pr   = std::promise<R>{};
  auto fut  = pr.get_future();

  FMPI_ASSERT(mpi::is_thread_main());
  auto const& config = Config::instance();

  if (config.main_core != core) {
    auto thread = std::thread(
        [task = function_t(std::move(task)), p = std::move(pr)]() mutable {
          detail::call_with_promise(std::move(task), p);
        });

    fmpi::pinThreadToCore(thread, core);
    thread.detach();
  } else {
    detail::call_with_promise(std::move(task), pr);
  }

  return fut;
}
}  // namespace fmpi

#endif
