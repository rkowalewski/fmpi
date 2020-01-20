#ifndef FMPI__CONTAINER__BOUNDED_BUFFER_H
#define FMPI__CONTAINER__BOUNDED_BUFFER_H

#include <boost/circular_buffer.hpp>
#include <condition_variable>
#include <list>
#include <mutex>

#include <fmpi/Debug.hpp>

namespace fmpi {

template <class Task, class Allocator = std::allocator<Task>>
class BoundedThreadsafeQueue {
 public:
  using container_type = boost::circular_buffer<Task, Allocator>;
  using size_type      = typename container_type::size_type;
  using value_type     = typename container_type::value_type;

  BoundedThreadsafeQueue(const BoundedThreadsafeQueue&) = delete;
  BoundedThreadsafeQueue& operator                      =(
      const BoundedThreadsafeQueue&);  // Disabled assign operator

  explicit BoundedThreadsafeQueue(size_type capacity)
    : container_(capacity) {
  }

  void push_front(const value_type& item) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_not_full_.wait(lock, [this]() { return is_not_full(); });
      container_.push_front(item);
    }
    cv_not_empty_.notify_one();
  }

  void push_front(value_type&& item) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_not_full_.wait(lock, [this]() { return is_not_full(); });
      container_.push_front(std::move(item));
    }
    cv_not_empty_.notify_one();
  }

  void pop_back(value_type& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_not_empty_.wait(lock, [this]() { return is_not_empty(); });
      value = std::move(container_.back());
      container_.pop_back();
    }  // automatically unlocks the mutex...
    cv_not_full_.notify_one();
  }

  template <class OutputIterator>
  void pop_back(OutputIterator it, std::size_t max) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_not_empty_.wait(lock, [this]() { return is_not_empty(); });
      auto const n     = std::min(max, container_.size());
      auto       first = std::make_reverse_iterator(std::end(container_));

      std::move(first, std::next(first, n), it);
      container_.erase_end(n);
    }  // automatically unlocks the mutex...
    cv_not_full_.notify_one();
  }

 private:
  bool is_not_empty() const {
    return container_.size() > 0;
  }
  bool is_not_full() const {
    return container_.size() < container_.capacity();
  }

  container_type          container_;
  std::mutex              mutex_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
};

template <class Task, class Allocator = std::allocator<Task>>
class ThreadsafeQueue {
 public:
  using container_type = std::list<Task, Allocator>;
  using size_type      = typename container_type::size_type;
  using value_type     = typename container_type::value_type;

  ThreadsafeQueue(const ThreadsafeQueue&) = delete;
  ThreadsafeQueue& operator=(const ThreadsafeQueue&) = delete;

  ThreadsafeQueue() = default;

  void push_front(const value_type& item) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      container_.push_front(item);
    }
    cv_not_empty_.notify_one();
  }

  void push_front(value_type&& item) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      container_.push_front(std::move(item));
      FMPI_DBG(container_.size());
    }
    cv_not_empty_.notify_one();
  }

  void pop_back(value_type& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this]() { return is_not_empty(); });
    value = std::move(container_.back());
    container_.pop_back();
  }

  template <class OutputIterator>
  void pop_back(OutputIterator it, std::size_t max) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this]() { return is_not_empty(); });
    auto const n     = std::min(max, container_.size());
    auto       first = std::make_reverse_iterator(std::end(container_));

    std::move(first, std::next(first, n), it);
    container_.erase_end(n);
  }

  bool empty() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return is_not_empty();
  }

 private:
  bool is_not_empty() const noexcept {
    return container_.size() > 0;
  }

  container_type          container_;
  std::mutex              mutex_;
  std::condition_variable cv_not_empty_;
};

}  // namespace fmpi
#endif
