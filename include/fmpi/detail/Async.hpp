#ifndef FMPI_DETAIL_ASYNC_HPP
#define FMPI_DETAIL_ASYNC_HPP

#include <future>

#include <fmpi/Utils.hpp>
#include <fmpi/detail/Capture.hpp>

namespace fmpi {

template <typename RET, typename FUNC, typename... ARGS>
constexpr void call_with_promise(
    fmpi::Capture<RET, FUNC, ARGS...>& callable, std::promise<RET>& pr) {
  pr.set_value(callable());
}

template <typename FUNC, typename... ARGS>
constexpr void call_with_promise(
    fmpi::Capture<void, FUNC, ARGS...>& callable, std::promise<void>& pr) {
  callable();
  pr.set_value();
}

template <typename R, typename F, typename... Ts>
inline std::future<R> async(int core, F&& f, Ts&&... params) {
  auto lambda =
      fmpi::makeCapture<R>(std::forward<F>(f), std::forward<Ts>(params)...);

  auto pr  = std::promise<R>{};
  auto fut = pr.get_future();

  auto thread =
      std::thread([fn = std::move(lambda), p = std::move(pr)]() mutable {
        call_with_promise(fn, p);
      });

  fmpi::pinThreadToCore(thread, core);
  thread.detach();

  return fut;
}
}  // namespace fmpi

#endif
