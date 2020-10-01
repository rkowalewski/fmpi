#ifndef FMPI_CONCURRENCY_SPSC_HPP
#define FMPI_CONCURRENCY_SPSC_HPP

#include <fmpi/concurrency/BufferedChannel.hpp>
#include <fmpi/util/Trace.hpp>

namespace fmpi {

template <class T>
class SPSCNChannel {
  using duration = typename steady_timer::duration;

 public:
  struct Stats {
    // consumer
    alignas(kCacheAlignment) duration dequeue_time{0};
    // producer
    alignas(kCacheAlignment) duration enqueue_time{0};
  };

  using value_type = T;
  using channel    = buffered_channel<value_type>;

  SPSCNChannel() = default;

  explicit SPSCNChannel(std::size_t n) noexcept
    : channel_(n)
    , high_watermark_(n)
    , count_(n) {
    FMPI_DBG("< SPSCNChannel(chan, n)");
    FMPI_DBG(task_count());
  }

  SPSCNChannel(SPSCNChannel const&) = delete;
  SPSCNChannel& operator=(SPSCNChannel const&) = delete;

  bool wait_dequeue(value_type& val) {
    steady_timer t_dequeue{stats_.dequeue_time};
    if (count_.load(std::memory_order_relaxed) == 0u) {
      return false;
    }
    val = channel_.value_pop();
    count_.fetch_sub(1, std::memory_order_release);
    return true;
  }

  template <class Rep, class Period>
  bool wait_dequeue(
      value_type& val, std::chrono::duration<Rep, Period> const& timeout) {
    steady_timer t_enqueue{stats_.dequeue_time};
    if (!count_.load(std::memory_order_relaxed)) {
      return false;
    }
    auto const ret     = channel_.pop(val, timeout);
    auto const success = ret == channel_op_status::success;
    count_.fetch_sub(success, std::memory_order_release);
    return success;
  }

  bool enqueue(value_type const& task) {
    steady_timer t_enqueue{stats_.enqueue_time};
    auto const   status = channel_.push(task);
    auto const   ret    = status == channel_op_status::success;

    if ((nproduced_ += ret) == high_watermark_) {
      // channel_.close();
    }

    return ret;
  }

  [[nodiscard]] bool done() const noexcept {
    return task_count() == 0U;
  }

  [[nodiscard]] std::size_t task_count() const noexcept {
    return count_.load(std::memory_order_acquire);
  }

  Stats statistics() const noexcept {
    FMPI_ASSERT(done());
    return stats_;
  }

  [[nodiscard]] std::size_t high_watermark() const noexcept {
    return high_watermark_;
  }

 private:
  channel channel_{0};

  // two separate write cachelines
  // L1: consumer
  // L2: producer
  Stats stats_{};

  // private access in producer
  std::size_t nproduced_{0};

  // shared read cacheline
  alignas(kCacheAlignment) std::size_t const high_watermark_{0};

  // shared write cacheline
  alignas(kCacheAlignment) std::atomic<std::size_t> count_{0};
};

}  // namespace fmpi

#endif
