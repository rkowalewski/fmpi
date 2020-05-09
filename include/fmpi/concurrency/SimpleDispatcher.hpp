#ifndef FMPI_CONCURRENCY_SIMPLEDISPATCHER_HPP
#define FMPI_CONCURRENCY_SIMPLEDISPATCHER_HPP

#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <queue>
#include <utility>

#include <fmpi/Config.hpp>
#include <fmpi/Function.hpp>
#include <fmpi/Message.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/concurrency/UnlockGuard.hpp>
#include <fmpi/mpi/TypeMapper.hpp>

namespace fmpi {

class SimpleDispatcher {
 public:
  using message = std::pair<Message, Message>;
  using task    = Function<void(void)>;

 public:
  SimpleDispatcher();
  ~SimpleDispatcher();

  // dispatch and copy
  std::future<mpi::return_code> dispatch(const message&);
  // dispatch and move
  std::future<mpi::return_code> dispatch(message&&);

  // Deleted operations
  SimpleDispatcher(const SimpleDispatcher&) = delete;
  SimpleDispatcher& operator=(const SimpleDispatcher&) = delete;
  SimpleDispatcher(SimpleDispatcher&&)                 = delete;
  SimpleDispatcher& operator=(SimpleDispatcher&&) = delete;

 private:
  std::mutex              lock_;
  std::thread             thread_;
  std::queue<task>        q_;
  std::condition_variable cv_;
  bool                    quit_ = false;

  void dispatch_thread_handler(void);
};

SimpleDispatcher::SimpleDispatcher() {
  thread_ = std::thread(&SimpleDispatcher::dispatch_thread_handler, this);
  auto const& config = Pinning::instance();
  pinThreadToCore(thread_, config.dispatcher_core);
}

SimpleDispatcher::~SimpleDispatcher() {
  // Signal to dispatch threads that it's time to wrap up
  {
    std::lock_guard<std::mutex> lock(lock_);
    quit_ = true;
  }
  cv_.notify_all();

  thread_.join();
}

std::future<mpi::return_code> SimpleDispatcher::dispatch(
    const message& message) {
  auto promise = std::make_unique<std::promise<mpi::return_code>>();

  auto future = promise->get_future();

  auto callable = task::make(
      [promise = std::move(promise), msg_pair = message]() mutable {
        auto& send = msg_pair.first;
        auto& recv = msg_pair.second;
        promise->set_value(MPI_Sendrecv(
            send.readable_buffer(),
            send.count(),
            send.type(),
            send.peer(),
            kTagBruck,
            recv.writable_buffer(),
            recv.count(),
            recv.type(),
            recv.peer(),
            kTagBruck,
            recv.comm(),
            MPI_STATUS_IGNORE));
      });
  {
    std::lock_guard<std::mutex> lock(lock_);

    q_.push(std::move(callable));
  }

  cv_.notify_all();

  return future;
}

std::future<mpi::return_code> SimpleDispatcher::dispatch(message&& message) {
  auto promise = std::make_unique<std::promise<mpi::return_code>>();

  auto future = promise->get_future();

  auto callable = task::make([promise  = std::move(promise),
                              msg_pair = std::move(message)]() mutable {
    auto& send = msg_pair.first;
    auto& recv = msg_pair.second;
    promise->set_value(MPI_Sendrecv(
        send.readable_buffer(),
        send.count(),
        send.type(),
        send.peer(),
        kTagBruck,
        recv.writable_buffer(),
        recv.count(),
        recv.type(),
        recv.peer(),
        kTagBruck,
        recv.comm(),
        MPI_STATUS_IGNORE));
  });
  {
    std::lock_guard<std::mutex> lock(lock_);

    q_.push(std::move(callable));
  }

  cv_.notify_all();

  return future;
}

void SimpleDispatcher::dispatch_thread_handler(void) {
  std::unique_lock<std::mutex> lock(lock_);

  do {
    // Wait until we have data or a quit signal
    cv_.wait(lock, [this] { return (q_.size() || quit_); });

    // after wait, we own the lock
    if (!quit_ && q_.size()) {
      auto op = std::move(q_.front());
      q_.pop();

      {
        UnlockGuard<std::mutex> ulg{lock_};

        op();
      }
    }
  } while (!quit_);
}
}  // namespace fmpi
#endif
