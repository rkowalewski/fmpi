#ifndef FMPI__DETAIL__COMM_STATE_H
#define FMPI__DETAIL__COMM_STATE_H

#include <list>

#include <fmpi/Debug.h>

#include <fmpi/Memory.h>
#include <fmpi/NumericRange.h>

namespace fmpi {
namespace detail {

template <class T, std::size_t NReqs>
class CommState {
  /// nongrowing vector without initialization
  using simple_vector =
      tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using size_type     = typename simple_vector::size_type;
  using iterator_type = typename simple_vector::iterator;

  // Buffer for up to 2 * NReqs occupied chunks
  static constexpr size_t MAX_FREE      = 2 * NReqs;
  static constexpr size_t MAX_COMPLETED = MAX_FREE;

  static_assert(MAX_FREE > 0, "buffer size must be > 0");

  template <std::size_t N>
  using small_vector = SmallVector<iterator_type, N * sizeof(iterator_type)>;

 public:
  explicit CommState(size_type blocksize)
    : m_occupied(typename small_vector<NReqs>::allocator{m_arena_occupied})
    , m_completed(
          typename small_vector<MAX_COMPLETED>::allocator{m_arena_completed})
    , m_blocksize(blocksize)
    , m_buffer(blocksize * MAX_FREE)
  {
    // here we explicitly resize the container
    m_occupied.resize(NReqs);

    // explicitly reserve memory prevent future memory reallocation
    m_completed.reserve(MAX_COMPLETED);

    fill_freelist();

    RTLX_ASSERT(blocksize > 0);
  }

  iterator_type receive_allocate(int key)
  {
    RTLX_ASSERT(0 <= key && std::size_t(key) < NReqs);

    // access last block remove it from stack
    auto* freeBlock = pop_freelist();

    if (!freeBlock) {
      FMPI_DBG(m_completed.size());
      throw std::bad_alloc();
    }

    m_occupied[key] = freeBlock;
    return freeBlock;
  }

  void receive_complete(int key)
  {
    RTLX_ASSERT(0 <= key && std::size_t(key) < NReqs);
    RTLX_ASSERT(m_completed.size() < MAX_COMPLETED);

    m_completed.push_back(m_occupied[key]);
    m_occupied[key] = nullptr;
  }

  void release_completed()
  {
    RTLX_ASSERT(m_completed.size() <= MAX_COMPLETED);

    for (auto&& b : m_completed) {
      push_freelist(b);
    }

    // 2) reset completed receives
    m_completed.clear();
  }

  std::size_t available_slots() const noexcept
  {
    return m_freelist.size();
  }

  auto const& completed_receives() const noexcept
  {
    return m_completed;
  }

 private:
  void push_freelist(T* block)
  {
    RTLX_ASSERT(block);

    m_freelist.push_front(block);
  }

  T* pop_freelist()
  {
    if (m_freelist.empty()) return nullptr;

    auto* block = m_freelist.front();

    m_freelist.pop_front();

    return block;
  }

  void fill_freelist()
  {
    RTLX_ASSERT(m_buffer.size());

    for (auto&& i : range<std::size_t>(1, MAX_FREE)) {
      auto b = std::prev(std::end(m_buffer), (i + 1) * m_blocksize);
      push_freelist(&*b);
    }
  }

 private:
  // occupied request slots
  typename small_vector<NReqs>::arena  m_arena_occupied{};
  typename small_vector<NReqs>::vector m_occupied{};

  // completed requests
  typename small_vector<MAX_COMPLETED>::arena  m_arena_completed{};
  typename small_vector<MAX_COMPLETED>::vector m_completed{};

  // freelist
  std::list<T*> m_freelist{};

  // buffer which contains both occupied and completed slots
  std::size_t   m_blocksize{};
  simple_vector m_buffer{};
};  // namespace detail

}  // namespace detail
}  // namespace fmpi

#endif
