#ifndef FMPI_CONCURRENCY_UNLOCKGUARD_HPP
#define FMPI_CONCURRENCY_UNLOCKGUARD_HPP

namespace fmpi {

template <typename Lockable>
struct UnlockGuard {
  explicit UnlockGuard(Lockable& mtx_)
    : mtx(mtx_) {
    mtx.unlock();
  }
  ~UnlockGuard() {
    mtx.lock();
  }
  UnlockGuard(UnlockGuard const&) = delete;
  UnlockGuard(UnlockGuard&&)      = delete;
  UnlockGuard& operator=(UnlockGuard const&) = delete;
  UnlockGuard& operator=(UnlockGuard&&) = delete;

 private:
  Lockable& mtx;
};
}  // namespace fmpi
#endif
