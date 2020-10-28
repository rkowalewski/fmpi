#ifndef OSU_HPP
#define OSU_HPP

#include <cstddef>
#include <fmpi/mpi/Environment.hpp>

#define WINDOW_SIZES \
  { 1, 2, 4, 8, 16, 32, 64, 128 }

#define WINDOW_SIZES_COUNT (8)

#define MPI_CHECK(stmt)                         \
  do {                                          \
    int mpi_errno = (stmt);                     \
    if (MPI_SUCCESS != mpi_errno) {             \
      fprintf(                                  \
          stderr,                               \
          "[%s:%d] MPI call failed with %d \n", \
          __FILE__,                             \
          __LINE__,                             \
          mpi_errno);                           \
      exit(EXIT_FAILURE);                       \
    }                                           \
    assert(MPI_SUCCESS == mpi_errno);           \
  } while (0)

constexpr std::size_t LARGE_MESSAGE_SIZE = 8192;
constexpr int         FIELD_WIDTH        = 18;
constexpr int         FLOAT_PRECISION    = 2;
constexpr int         WINDOW_SIZE_LARGE  = 64;

struct Params {
  Params()                       = default;
  unsigned int iterations        = 10000;
  unsigned int iterations_large  = 1000;
  unsigned int warmups           = 100;
  unsigned int warmups_large     = 10;
  std::size_t  smin              = 0;
  std::size_t  smax              = 1ul << 22;  // 4k
  bool         window_varied     = true;
  int          window_size       = WINDOW_SIZE_LARGE;
  int          window_size_large = WINDOW_SIZE_LARGE;
  int          pairs;
};

extern struct Params options;

bool read_input(int argc, char* argv[]);

int allocate_memory_pt2pt(char** sbuf, char** rbuf, int rank);

int  allocate_memory_pt2pt_mul(char** sbuf, char** rbuf, int rank, int pairs);
void free_memory_pt2pt_mul(void* sbuf, void* rbuf, int rank, int pairs);

void set_buffer_pt2pt(void* buffer, int rank, int data, size_t size);

void     print_topology(mpi::Context const& ctx, std::size_t nhosts);
uint32_t num_nodes(mpi::Context const& comm);

#endif
