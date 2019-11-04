#ifndef FMPI__DETAIL__COMM_STATE_H
#define FMPI__DETAIL__COMM_STATE_H

#include <fmpi/Debug.h>

#include <fmpi/NumericRange.h>

namespace fmpi::detail {

template <class T, std::size_t NReqs>
class CommState {
  /// nongrowing vector without initialization
  using simple_vector =
      tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using size_type     = typename simple_vector::size_type;
  using iterator_type = typename simple_vector::iterator;
  using iterator_pair = std::pair<iterator_type, iterator_type>;

  // Buffer for up to 2 * NReqs pending chunks
  static constexpr size_t MAX_FREE_CHUNKS      = 2 * NReqs;
  static constexpr size_t MAX_COMPLETED_CHUNKS = MAX_FREE_CHUNKS;

  template <class _T, std::size_t N>
  using stack_arena = tlx::StackArena<N * sizeof(_T)>;

  template <class _T, std::size_t N>
  using stack_allocator = tlx::StackAllocator<_T, N * sizeof(_T)>;

  template <class _T, std::size_t N>
  using stack_vector = std::vector<_T, stack_allocator<_T, N>>;

 public:
  explicit CommState(size_type blocksize)
    : m_completed(
          0,
          iterator_pair{},
          stack_allocator<iterator_pair, MAX_COMPLETED_CHUNKS>{
              m_arena_completed})
    , m_freelist(stack_vector<iterator_pair, MAX_FREE_CHUNKS>{
          0,
          iterator_pair{},
          stack_allocator<iterator_pair, MAX_FREE_CHUNKS>{m_arena_freelist}})
    , m_buffer(blocksize * MAX_FREE_CHUNKS)
  {
    std::fill(std::begin(m_pending), std::end(m_pending), iterator_pair{});

    for (auto&& block : fmpi::range<int>(MAX_FREE_CHUNKS - 1, -1, -1)) {
      auto f = std::next(std::begin(m_buffer), block * blocksize);
      auto l = std::next(f, blocksize);
      m_freelist.push(std::make_pair(f, l));
    }
  }

  iterator_type receive_allocate(int key)
  {
    RTLX_ASSERT(m_arena_freelist.size() > 0);
    RTLX_ASSERT(0 <= key && std::size_t(key) < NReqs);
    RTLX_ASSERT(!m_freelist.empty() && m_freelist.size() <= MAX_FREE_CHUNKS);

    // access last block remove it from stack
    auto freeBlock = m_freelist.top();
    m_freelist.pop();

    m_pending[key] = freeBlock;
    return freeBlock.first;
  }

  void receive_complete(int key)
  {
    RTLX_ASSERT(m_arena_completed.size() > 0);
    RTLX_ASSERT(0 <= key && std::size_t(key) < NReqs);
    RTLX_ASSERT(m_completed.size() < MAX_COMPLETED_CHUNKS);

    auto block = m_pending[key];
    m_completed.push_back(block);
  }

  void release_completed()
  {
    RTLX_ASSERT(m_arena_freelist.size() > 0);
    RTLX_ASSERT(m_completed.size() <= MAX_COMPLETED_CHUNKS);

    for (auto const& completed : m_completed) {
      m_freelist.push(completed);
    }

    // 2) reset completed receives
    m_completed.clear();

    RTLX_ASSERT(!m_freelist.empty() && m_freelist.size() <= MAX_FREE_CHUNKS);
  }

  std::size_t available_slots() const noexcept
  {
    return m_freelist.size();
  }

  stack_vector<iterator_pair, MAX_COMPLETED_CHUNKS> const&
  completed_receives() const noexcept
  {
    return m_completed;
  }

 private:
  stack_arena<iterator_pair, MAX_COMPLETED_CHUNKS>  m_arena_completed{};
  stack_vector<iterator_pair, MAX_COMPLETED_CHUNKS> m_completed{};

  stack_arena<iterator_pair, MAX_FREE_CHUNKS> m_arena_freelist{};
  std::stack<iterator_pair, stack_vector<iterator_pair, MAX_FREE_CHUNKS>>
      m_freelist{};

  std::array<iterator_pair, NReqs> m_pending{};

  simple_vector m_buffer{};
};

}  // namespace fmpi::detail

#endif
