#ifndef FMPI_MEMORY_BLOCKALLOCATOR_HPP
#define FMPI_MEMORY_BLOCKALLOCATOR_HPP

#include <cstddef>
#include <cstdint>

// POSSIBLE ISSUES:
// - Since the size of each allocation is NUM_BLOCKS * BLOCK_SIZE +
// _header_size, if the user allocates
//  2 blocks the allocation size is 8208 (BLOCK_SIZE = 4096, _header_size =
//  16) Later, if the user frees the 2 blocks and allocates a new one,
//  BLOCK_SIZE bytes (4096 bytes), might be wasted because the new hole is not
//  large enough to store a block plus its header
//
//  Possible solutions:
//   1 - Use a different algorithm to find the best fit free block,
//    that prevents (or at least minimizes) the issue
//   2 - Always allocate NUM_BLOCKS * (BLOCK_SIZE + _header_size) bytes so the
//   issue doesnt.
//    (This solution causes a fixed ammount of memory to be reserved to header
//    sizes)
//     Eg: BLOCK_SIZE = 4096, _header_size = 16, total_memory = 2GiB
//      ~8MiB reserved for headers (~0.004% of total memory reserved)

namespace fmpi {

class BlockAllocator {
 public:
  BlockAllocator(
      size_t size, void* start, size_t block_size, std::size_t alignment);
  ~BlockAllocator();

  void* allocate(size_t size, std::size_t alignment);
  void  deallocate(void* p);

  size_t getBlockSize() const;
  size_t getAlignment() const;

 private:
  struct BlockHeader {
    size_t       size;
    BlockHeader* next_free_block;
  };

  BlockHeader* _free_blocks;

  // size_t _size;
  size_t _block_size;
  size_t _header_size;  // sizeof(BlockHeader) + adjustment to keep blocks
                        // properly aligned
  std::size_t _used_memory;
  std::size_t _num_allocations;
  std::size_t _alignment;
  std::size_t _initial_adjustment;

  BlockAllocator(const BlockAllocator&) = delete;
  BlockAllocator& operator=(const BlockAllocator&) = delete;
};
}  // namespace fmpi
#endif
