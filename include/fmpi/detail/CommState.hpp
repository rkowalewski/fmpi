#ifndef FMPI_DETAIL_COMMSTATE_HPP
#define FMPI_DETAIL_COMMSTATE_HPP

#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <list>
#include <tlx/simple_vector.hpp>

#include <boost/lockfree/spsc_queue.hpp>

namespace fmpi {
namespace detail {

template <class T>
struct SlidingReqWindow {
 private:
  /// nongrowing vector without initialization
  using simple_vector =
      tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using iterator  = typename simple_vector::iterator;
  using iter_pair = std::pair<iterator, iterator>;

  simple_vector storage_{};

 public:
  SlidingReqWindow(std::size_t winsize, std::size_t blocksize)
    : storage_(2 * winsize * blocksize)
    , winsize_(winsize)
    , recvbuf_(storage_.begin())
    , mergebuf_(std::next(recvbuf_, storage_.size() / 2))
  {
    // We explitly reserve one additional chunk for the local portion
    pending_.reserve(winsize + 1);
    ready_.reserve(winsize + 1);
  }

  [[nodiscard]] auto winsize() const noexcept
  {
    return winsize_;
  }

  void buffer_swap()
  {
    // swap chunks
    std::swap(pending_, ready_);
    // swap buffers
    std::swap(recvbuf_, mergebuf_);
  }

  [[nodiscard]] auto rbuf() const noexcept -> iterator
  {
    return recvbuf_;
  }

  [[nodiscard]] auto mergebuf() const noexcept -> iterator
  {
    return mergebuf_;
  }

  auto& pending_pieces() noexcept
  {
    return pending_;
  }

  [[nodiscard]] auto const& pending_pieces() const noexcept
  {
    return pending_;
  }

  auto& ready_pieces() noexcept
  {
    return ready_;
  }

  [[nodiscard]] auto const& ready_pieces() const noexcept
  {
    return ready_;
  }

  std::size_t            winsize_{};
  iterator               recvbuf_{};
  iterator               mergebuf_{};
  std::vector<iter_pair> pending_{};
  std::vector<iter_pair> ready_{};
};  // namespace detail

}  // namespace detail
}  // namespace fmpi

#endif
