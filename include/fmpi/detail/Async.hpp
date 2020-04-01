#ifndef FMPI_DETAIL_ASYNC_HPP
#define FMPI_DETAIL_ASYNC_HPP

#include <future>

#include <fmpi/common/Porting.hpp>
#include <fmpi/detail/Capture.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Timer.hpp>

namespace fmpi {

namespace detail {

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
}  // namespace detail

template <typename R, typename F, typename... Ts>
inline std::future<R> async(int core, F&& f, Ts&&... params) {
  auto lambda =
      fmpi::makeCapture<R>(std::forward<F>(f), std::forward<Ts>(params)...);

  auto pr  = std::promise<R>{};
  auto fut = pr.get_future();

  FMPI_ASSERT(mpi::is_thread_main());
  auto const& config = Config::instance();

  if (config.main_core != core) {
#if defined(NDEBUG)
    auto thread =
        std::thread([fn = std::move(lambda), p = std::move(pr)]() mutable {
            detail::call_with_promise(fn, p);
        });
#else
    auto thread = std::thread(
        [fn = std::move(lambda), p = std::move(pr), core]() mutable {
          auto const cpu = sched_getcpu();
          FMPI_DBG(std::make_pair(cpu, core));
          //  FMPI_ASSERT(cpu == core);

          detail::call_with_promise(fn, p);
        });
#endif

    fmpi::pinThreadToCore(thread, core);
    thread.detach();
  } else {
    detail::call_with_promise(lambda, pr);
  }

  return fut;
}
}  // namespace fmpi

#endif
