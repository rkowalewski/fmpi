#ifndef FMPI_MPI_ENVIRONMENT_HPP
#define FMPI_MPI_ENVIRONMENT_HPP

#include <cstdint>

#include <mpi.h>

#include <fmpi/mpi/Rank.hpp>

namespace mpi {

class Context {
 public:
  using size_type = std::uint32_t;

 public:
  explicit Context(MPI_Comm comm);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context() = default;

  [[nodiscard]] Rank rank() const noexcept;

  [[nodiscard]] size_type size() const noexcept;

  [[nodiscard]] MPI_Comm mpiComm() const noexcept;

  static Context& world();

 private:
  size_type m_size{};
  Rank      m_rank{};
  MPI_Comm  m_comm{MPI_COMM_NULL};
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

auto splitSharedComm(Context const& baseComm) -> Context;

}  // namespace mpi

#endif
