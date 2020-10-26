#include <boost/pool/pool_alloc.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/detail/Assert.hpp>
#include <fmpi/memory/HeapAllocator.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>

namespace fmpi {

#if 0
namespace detail {

class SmallRequestPool {
 public:
  SmallRequestPool(std::size_t n)
    : alloc_(static_cast<uint16_t>(n)) {
  }

  std::unique_ptr<MPI_Request, RequestDelete> allocate() {
    return std::unique_ptr<MPI_Request, RequestDelete>(
        alloc_.allocate(1), RequestDelete{alloc_});
  }

 private:
  fmpi::HeapAllocator<MPI_Request> alloc_;
};

}  // namespace detail
#endif

// static detail::SmallRequestPool s_request_pool{initial_req_cap};

// static ThreadAllocator<detail::future_shared_state> s_future_alloc;
static constexpr std::uint16_t initial_req_cap = 1000;
static HeapAllocator<std::shared_ptr<detail::future_shared_state>>
    s_future_alloc{initial_req_cap};

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
}

std::shared_ptr<collective_future::simple_message_queue> const&
collective_future::allocate_queue(std::size_t n) {
  partials_ = std::make_shared<simple_message_queue>(n);
  return partials_;
}

std::shared_ptr<collective_future::simple_message_queue> const&
collective_future::arrival_queue() {
  FMPI_ASSERT(partials_);
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

bool collective_future::is_deferred() const noexcept {
  return valid() && sptr_->is_deferred();
}

mpi::return_code collective_future::get() {
  wait();
  auto sptr = std::move(sptr_);
  return sptr->get_value_assume_ready();
}

MPI_Request& collective_future::native_handle() noexcept {
  FMPI_ASSERT(valid());
  return sptr_->native_handle();
}

const MPI_Request& collective_future::native_handle() const noexcept {
  FMPI_ASSERT(valid());
  return sptr_->native_handle();
}

namespace detail {
future_shared_state::future_shared_state(state s)
  : state_(s) {
}

void future_shared_state::wait() {
  if (is_deferred() and not ready_) {
    auto ret = MPI_Wait(&mpi_handle_, MPI_STATUS_IGNORE);
    value_.emplace(ret);
    // ready_ = true;
  } else {
    std::unique_lock<std::mutex> lk(mtx_);
    while (not ready_) {
      cv_.wait(lk);
    }
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

collective_future make_ready_future(mpi::return_code u) {
  auto state = std::make_shared<detail::future_shared_state>();
  state->set_value(u);
  return collective_future{std::move(state)};
}

collective_future make_mpi_future() {
  auto thread_safe_alloc =
      fmpi::ThreadAllocator<detail::future_shared_state>{};
  auto sp = std::allocate_shared<detail::future_shared_state>(
      thread_safe_alloc, detail::future_shared_state::state::deferred);

  return collective_future{std::move(sp)};
}
}  // namespace fmpi
