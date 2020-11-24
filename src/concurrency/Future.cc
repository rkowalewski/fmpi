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
collective_future::allocate_queue(std::size_t cap, std::size_t expected) {
  FMPI_ASSERT(not partials_);
  partials_ = std::make_shared<simple_message_queue>(cap);

  expected_ = (expected == 0) ? cap : expected;

  return partials_;
}

std::shared_ptr<collective_future::simple_message_queue> const&
collective_future::arrival_queue() {
  FMPI_ASSERT(partials_);
  return partials_;
}

collective_future::~collective_future() {
  FMPI_DBG(valid());
  if (valid()) {
    wait();
  }
}
void collective_future::wait() {
  FMPI_ASSERT(valid());
  sptr_->wait();
  FMPI_DBG("notify");
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

std::size_t collective_future::expected() const noexcept {
  return expected_;
}

mpi::return_code collective_future::get() {
  wait();
  auto sptr = std::move(sptr_);
  return sptr->get_value_assume_ready();
}

std::vector<MPI_Request>& collective_future::native_handles() noexcept {
  FMPI_ASSERT(valid());
  return sptr_->native_handles();
}

std::vector<MPI_Request> const& collective_future::native_handles()
    const noexcept {
  FMPI_ASSERT(valid());
  return sptr_->native_handles();
}

namespace detail {
future_shared_state::future_shared_state(state s)
  : state_(s) {
}

void future_shared_state::wait() {
  if (is_deferred()) {
    auto const size = mpi_handles_.size();
    FMPI_ASSERT(size < std::numeric_limits<int>::max());
    auto ret = MPI_Waitall(
        static_cast<int>(size), mpi_handles_.data(), MPI_STATUSES_IGNORE);
    FMPI_ASSERT(ret == MPI_SUCCESS);
    value_.emplace(ret);
  } else {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this]() { return unsafe_is_ready(); });
  }
}

void future_shared_state::set_value(mpi::return_code result) {
  {
    std::lock_guard lk(mtx_);
    value_.emplace(result);
  }

  cv_.notify_all();

  // m_then.swap(continuation);

  // if (continuation) {
  //  continuation();
  //}
}

bool future_shared_state::is_ready() const {
  std::lock_guard<std::mutex> lg{mtx_};
  return unsafe_is_ready();
}

bool future_shared_state::unsafe_is_ready() const noexcept {
  return value_.has_value();
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

collective_future make_mpi_future(std::size_t n) {
  auto sp = std::make_shared<detail::future_shared_state>(
      detail::future_shared_state::state::deferred);

  sp->native_handles().resize(n, MPI_REQUEST_NULL);

  return collective_future{std::move(sp)};
}
}  // namespace fmpi
