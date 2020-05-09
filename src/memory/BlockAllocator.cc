#include <fmpi/Debug.hpp>
#include <fmpi/memory/BlockAllocator.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>

namespace fmpi {

BlockAllocator::BlockAllocator(
    size_t size, void* start, size_t block_size, std::size_t alignment)
  : _block_size(block_size)
  , _used_memory(0)
  , _num_allocations(0)
  , _alignment(alignment) {
  // Implementation is simplified by making these two assumptions
  FMPI_ASSERT(_alignment >= alignof(BlockHeader));
  FMPI_ASSERT(_block_size % _alignment == 0);

  FMPI_ASSERT(size > 0 && start != nullptr);

  _header_size = sizeof(BlockHeader);

  auto const rem = sizeof(BlockHeader) % _alignment;

  if (rem != 0) _header_size += _alignment - rem;

  _initial_adjustment = detail::alignForwardAdjustment(start, _alignment);

  _free_blocks = (BlockHeader*)detail::add(start, _initial_adjustment);

  FMPI_ASSERT(detail::isAligned(_free_blocks));

  _free_blocks->size            = size - _initial_adjustment - _header_size;
  _free_blocks->next_free_block = nullptr;

  _used_memory += _initial_adjustment + _header_size;

  /*
  size_t k = BLOCK_SIZE + sizeof(AllocationHeader);

  //Choose proper alignment
  std::size_t alignment = ALIGNMENT > __alignof(AllocationHeader) ? ALIGNMENT
  :
  __alignof(AllocationHeader);

  std::uint8_t adjustment = detail::alignForwardAdjustment(start,
  alignment);

  //Align memory
  size -= adjustment;
  start = detail::add(start, adjustment);

  size_t num_blocks = size / k;

  FMPI_ASSERT(num_blocks > 0);
  */
}

BlockAllocator::~BlockAllocator() {
  _used_memory -= _initial_adjustment + _header_size;
}

void* BlockAllocator::allocate(size_t size, std::size_t alignment) {
  FMPI_ASSERT(size % _block_size == 0);
  FMPI_ASSERT(size > 0);
  FMPI_ASSERT(alignment == _alignment || alignment == 0);

  BlockHeader* prev_free_block = nullptr;
  BlockHeader* free_block      = _free_blocks;

  BlockHeader* best_fit_prev = nullptr;
  BlockHeader* best_fit      = nullptr;

  // Find best fit
  while (free_block != nullptr) {
    // size_t total_size = size + _header_size;

    // If its an exact match use this free block
    if (free_block->size == size) {
      best_fit_prev = prev_free_block;
      best_fit      = free_block;
      // best_fit_adjustment = adjustment;
      // best_fit_total_size = total_size;

      break;
    }

    // If its a better fit switch
    if (free_block->size > size &&
        (best_fit == nullptr || free_block->size < best_fit->size)) {
      best_fit_prev = prev_free_block;
      best_fit      = free_block;
      // best_fit_adjustment = adjustment;
      // best_fit_total_size = total_size;
    }

    prev_free_block = free_block;
    free_block      = free_block->next_free_block;
  }

  if (best_fit == nullptr) return nullptr;

  // If allocations in the remaining memory will be impossible
  if (best_fit->size - size < _header_size + _block_size) {
    // Increase allocation size instead of creating a new block
    size = best_fit->size;

    if (best_fit_prev != nullptr)
      best_fit_prev->next_free_block = best_fit->next_free_block;
    else
      _free_blocks = best_fit->next_free_block;
  } else {
    // Else create a new block containing remaining memory
    auto* new_block = (BlockHeader*)(detail::add(best_fit + 1, size));
    new_block->size        = best_fit->size - size - _header_size;
    new_block->next_free_block = best_fit->next_free_block;

    if (best_fit_prev != nullptr)
      best_fit_prev->next_free_block = new_block;
    else
      _free_blocks = new_block;

    _used_memory += _header_size;
  }

  void* block = detail::add(best_fit, _header_size);

  best_fit->size            = size;
  best_fit->next_free_block = nullptr;

  _used_memory += size;
  _num_allocations++;

  FMPI_ASSERT(detail::isAligned(block, _alignment));

  return block;
}

void BlockAllocator::deallocate(void* p) {
  FMPI_ASSERT(p != nullptr);

  auto* header = (BlockHeader*)detail::sub(p, _header_size);

  auto block_start = reinterpret_cast<std::uintptr_t>(header);
  size_t         block_size  = header->size;
  std::uintptr_t block_end   = block_start + block_size + _header_size;

  BlockHeader* prev_free_block = nullptr;
  BlockHeader* free_block      = _free_blocks;

  while (free_block != nullptr) {
    if ((std::uintptr_t)free_block >= block_end) break;

    prev_free_block = free_block;
    free_block      = free_block->next_free_block;
  }

  if (prev_free_block == nullptr) {
    prev_free_block                  = (BlockHeader*)block_start;
    prev_free_block->size            = block_size;
    prev_free_block->next_free_block = _free_blocks;

    _free_blocks = prev_free_block;

    _used_memory -= block_size;
  } else if (
      (std::uintptr_t)prev_free_block + _header_size +
          prev_free_block->size ==
      block_start) {
    prev_free_block->size += _header_size + block_size;

    _used_memory -= _header_size + block_size;
  } else {
    header->next_free_block          = prev_free_block->next_free_block;
    prev_free_block->next_free_block = header;

    prev_free_block = header;

    _used_memory -= block_size;
  }

  FMPI_ASSERT(prev_free_block != nullptr);

  if ((std::uintptr_t)prev_free_block + _header_size +
          prev_free_block->size ==
      (std::uintptr_t)prev_free_block->next_free_block) {
    _used_memory -= _header_size;

    prev_free_block->size +=
        _header_size + prev_free_block->next_free_block->size;
    prev_free_block->next_free_block =
        prev_free_block->next_free_block->next_free_block;
  }

  _num_allocations--;
}

size_t BlockAllocator::getBlockSize() const {
  return _block_size;
}

size_t BlockAllocator::getAlignment() const {
  return _alignment;
}
}  // namespace fmpi
