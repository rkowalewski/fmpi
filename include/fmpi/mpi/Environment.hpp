#ifndef FMPI_MPI_ENVIRONMENT_HPP
#define FMPI_MPI_ENVIRONMENT_HPP

#include <mpi.h>

#include <atomic>
#include <fmpi/memory/HeapAllocator.hpp>
#include <fmpi/mpi/Rank.hpp>

namespace mpi {

class Context;
Context splitSharedComm(Context const& baseComm);

class Context {
  friend Context splitSharedComm(Context const& baseComm);

  Context(MPI_Comm comm, bool free_self);
  void free();

  static constexpr const uint16_t req_pool_cap = 100;

 public:
  using size_type      = std::uint32_t;

  explicit Context(MPI_Comm comm);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context();

  [[nodiscard]] constexpr Rank rank() const noexcept {
    return m_rank;
  }

  [[nodiscard]] constexpr size_type size() const noexcept {
    return m_size;
  }

  [[nodiscard]] constexpr explicit operator MPI_Comm() const noexcept {
    return mpiComm();
  }

  [[nodiscard]] constexpr MPI_Comm mpiComm() const noexcept {
    return m_comm;
  }

  [[nodiscard]] constexpr MPI_Group mpiGroup() const noexcept {
    return m_group;
  }

  int32_t collectiveTag() const;

  static Context const& world();

 private:
  MPI_Comm                    m_comm{MPI_COMM_NULL};
  MPI_Group                   m_group{};
  size_type                   m_size{};
  Rank                        m_rank{};
  mutable std::atomic_int32_t m_collective_tag{};
  bool                        m_free_self{false};
};

enum class ThreadLevel : int
{
  Single     = MPI_THREAD_SINGLE,
  Funneled   = MPI_THREAD_FUNNELED,
  Serialized = MPI_THREAD_SERIALIZED,
  Multiple   = MPI_THREAD_MULTIPLE
};

bool initialize(int* argc, char*** argv, ThreadLevel level);
void finalize();
bool is_thread_main();

}  // namespace mpi

#endif
