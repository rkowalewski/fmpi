#ifndef FMPI_CONCURRENCY_SIMPLE_CONCURRENT_DEQUE_HPP
#define FMPI_CONCURRENCY_SIMPLE_CONCURRENT_DEQUE_HPP

#include <condition_variable>
#include <deque>
#include <mutex>

namespace fmpi {

//! \brief A templated *thread-safe* collection based on dequeue
//!
template <typename T>
class SimpleConcurrentDeque {
 public:
  using value_type = T;

 public:
  SimpleConcurrentDeque() noexcept = default;

  //! \brief Emplaces a new instance of T in front of the deque
  template <typename... Args>
  void emplace_front(Args&&... args) {
    addData_protected(
        [&] { collection_.emplace_front(std::forward<Args>(args)...); });
  }

  //! \brief Emplaces a new instance of T at the back of the deque
  template <typename... Args>
  void emplace_back(Args&&... args) {
    addData_protected(
        [&] { collection_.emplace_back(std::forward<Args>(args)...); });
  }

  void push_back(const T& v) {
    addData_protected([&] { collection_.push_back(v); });
  }

  void push_back(T&& v) {
    addData_protected([&] { collection_.push_back(v); });
  }

  //! \brief Clears the deque
  void clear(void) {
    std::lock_guard<std::mutex> lock(mtx_);
    collection_.clear();
  }

  //! \brief Returns the front element and removes it from the collection
  //!
  //!        No exception is ever returned as we garanty that the deque is not
  //!        empty before trying to return data.
  T pop_front(void) noexcept {
    std::unique_lock<std::mutex> lock{mtx_};
    cond_.wait(lock, [this]() { return not collection_.empty(); });
    auto elem = std::move(collection_.front());
    collection_.pop_front();
    return elem;
  }

  template <class OutputIterator>
  OutputIterator pop_all(OutputIterator it, std::size_t& n) {
    std::lock_guard<std::mutex> lock{mtx_};
    n        = collection_.size();
    auto ret = std::move(std::begin(collection_), std::end(collection_), it);

    collection_.clear();
    return ret;
  }

 private:
  //! \brief Protects the deque, calls the provided function and notifies the
  //! presence of new data \param The concrete operation to be used. It MUST
  //! be an operation which will add data to the deque,
  //!        as it will notify that new data are available!
  template <class F>
  void addData_protected(F&& fct) {
    {
      std::lock_guard<std::mutex> lock{mtx_};
      fct();
    }

    cond_.notify_one();
  }

  std::deque<T> collection_;      ///< Concrete, not thread safe, storage.
  std::mutex    mtx_;             ///< Mutex protecting the concrete storage
  std::condition_variable cond_;  ///< Condition used to notify that
                                  ///< new data are available.
};
}  // namespace fmpi

#endif
