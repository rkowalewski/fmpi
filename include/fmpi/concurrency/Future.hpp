#ifndef FMPI_CONCURRENCY_FUTURE_HPP
#define FMPI_CONCURRENCY_FUTURE_HPP

#include <atomic>
#include <fmpi/Message.hpp>
#include <fmpi/concurrency/SimpleConcurrentDeque.hpp>
#include <memory>
#include <optional>

namespace fmpi {

namespace detail {

class future_shared_state {
  /// Status Information
  std::mutex                      mtx_;
  std::condition_variable         cv_;
  std::atomic_bool                ready_{false};
  std::optional<mpi::return_code> value_;

 public:
  void             wait();
  void             set_value(mpi::return_code result);
  [[nodiscard]] bool is_ready() const noexcept;
  mpi::return_code get_value_assume_ready() noexcept;
};

}  // namespace detail

class collective_future;

class collective_promise {
  std::shared_ptr<detail::future_shared_state> sptr_;

 public:
  collective_promise();
  collective_promise(const collective_promise&)     = delete;
  collective_promise(collective_promise&&) noexcept = default;
  collective_promise& operator=(collective_promise&& rhs) noexcept;
  collective_promise& operator=(const collective_promise&) = delete;
  ~collective_promise();
  void swap(collective_promise& rhs) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool is_ready() const noexcept;

  void set_value(mpi::return_code res);

  collective_future get_future();
};

class collective_future {
  friend class collective_promise;
  friend collective_future make_ready_collective_future(mpi::return_code);

  using simple_message_queue = SimpleConcurrentDeque<Message>;

  std::shared_ptr<detail::future_shared_state> sptr_;
  // Partial arrivals
  std::shared_ptr<simple_message_queue> partials_;

  explicit collective_future(std::shared_ptr<detail::future_shared_state> p);

 public:
  collective_future() noexcept                    = default;
  collective_future(const collective_future&)     = delete;
  collective_future(collective_future&&) noexcept = default;
  collective_future& operator=(collective_future&&) noexcept = default;
  collective_future& operator=(const collective_future&) = delete;
  ~collective_future();

  void swap(collective_future& rhs);

  const std::shared_ptr<simple_message_queue>& arrival_queue();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool is_ready() const noexcept;
  void             wait();
  mpi::return_code get();
};

}  // namespace fmpi
#endif
