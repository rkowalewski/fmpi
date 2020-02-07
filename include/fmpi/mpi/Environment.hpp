#ifndef FMPI_MPI_ENVIRONMENT_HPP
#define FMPI_MPI_ENVIRONMENT_HPP

#include <mpi.h>

#include <cstdint>

#include <fmpi/mpi/Rank.hpp>

namespace mpi {

class Context {
 public:
  using size_type = std::uint32_t;

 public:
  explicit Context(MPI_Comm comm);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  [[nodiscard]] auto rank() const noexcept -> Rank;

  [[nodiscard]] auto size() const noexcept -> size_type;

  [[nodiscard]] auto mpiComm() const noexcept -> MPI_Comm;

 private:
  size_type m_size{};
  Rank      m_rank{};
  MPI_Comm  m_comm{MPI_COMM_NULL};
};

bool is_thread_main();

std::string processor_name();

auto splitSharedComm(Context const& baseComm) -> Context;

}  // namespace mpi

#endif
