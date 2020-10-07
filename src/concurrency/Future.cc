#include <fmpi/concurrency/Future.hpp>
#include <fmpi/detail/Assert.hpp>

namespace fmpi {

collective_promise::collective_promise() {
  sptr_ = std::make_shared<detail::future_shared_state>();
}

collective_promise& collective_promise::operator=(
    collective_promise&& rhs) noexcept {
  collective_promise(std::move(rhs)).swap(*this);
  return *this;
}

void collective_promise::swap(collective_promise& rhs) noexcept {
  sptr_.swap(rhs.sptr_);
}

collective_promise::~collective_promise() {
  FMPI_ASSERT(not valid() || is_ready());
}

bool collective_promise::valid() const noexcept {
  return sptr_ != nullptr;
}
bool collective_promise::is_ready() const noexcept {
  return valid() && sptr_->is_ready();
}

void collective_promise::set_value(mpi::return_code res) {
  FMPI_ASSERT(valid());
  sptr_->set_value(std::move(res));
}

collective_future collective_promise::get_future() {
  FMPI_ASSERT(valid());
  FMPI_ASSERT(sptr_.use_count() == 1);
  return collective_future(sptr_);
}

collective_future::collective_future(
    std::shared_ptr<detail::future_shared_state> p)
  : sptr_(std::move(p)) {
  partials_ = std::make_shared<simple_message_queue>();
}

std::shared_ptr<collective_future::simple_message_queue> const&
collective_future::arrival_queue() {
  return partials_;
}

collective_future::~collective_future() {
  if (valid()) {
    wait();
  }
}
void collective_future::wait() {
  FMPI_ASSERT(valid());
  sptr_->wait();
}

bool collective_future::valid() const noexcept {
  return sptr_ != nullptr;
}

bool collective_future::is_ready() const noexcept {
  return valid() && sptr_->is_ready();
}

mpi::return_code collective_future::get() {
  wait();
  auto sptr = std::move(sptr_);
  return sptr->get_value_assume_ready();
}

collective_future make_ready_future(mpi::return_code u) {
  auto state = std::make_shared<detail::future_shared_state>();
  state->set_value(u);
  return collective_future{state};
}  // namespace fmpi

namespace detail {

void future_shared_state::wait() {
  std::unique_lock<std::mutex> lk(mtx_);

  while (not ready_) {
    cv_.wait(lk);
  }
}

void future_shared_state::set_value(mpi::return_code result) {
  FMPI_ASSERT(!ready_);
  FMPI_ASSERT(!value_);
  value_.emplace(result);

  {
    std::lock_guard lk(mtx_);
    ready_ = true;
    // m_then.swap(continuation);
    cv_.notify_all();
  }

  // if (continuation) {
  //  continuation();
  //}
}

bool future_shared_state::is_ready() const noexcept {
  return ready_.load(std::memory_order_relaxed);
}

mpi::return_code future_shared_state::get_value_assume_ready() noexcept {
  FMPI_ASSERT(value_);
  return *value_;
}
}  // namespace detail
}  // namespace fmpi
