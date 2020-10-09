#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <chrono>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <iosfwd>
#include <string>

namespace benchmark {

namespace detail {
using namespace std::chrono_literals;
static constexpr auto five_seconds = 5s;
}  // namespace detail

struct Params {
 public:
  Params() noexcept;
  unsigned int              niters{};
  unsigned int              warmups{};
  std::size_t               smin       = 1u << 5;  // 32 bytes
  std::size_t               smax       = 1u << 7;  // 128 bytes
  uint32_t                  pmin       = 1u << 1;
  uint32_t                  pmax       = pmin;
  std::chrono::microseconds time_limit = detail::five_seconds;

  std::string pattern{};     // pattern for algorithm selection
  bool        check{false};  // validate correctness
};

#if 0
struct CollectiveArgs {
 public:
#if 0
  constexpr CollectiveArgs(
      const void*         sendbuf_,
      std::size_t         sendcount_,
      MPI_Datatype        sendtype_,
      void*               recvbuf_,
      std::size_t         recvcount_,
      MPI_Datatype        recvtype_,
      mpi::Context const& comm_)
    : sendbuf(sendbuf_)
    , sendcount(sendcount_)
    , sendtype(sendtype_)
    , recvbuf(recvbuf_)
    , recvcount(recvcount_)
    , recvtype(recvtype_)
    , comm(comm_) {
  }
#endif

  template <class T>
  constexpr CollectiveArgs(
      const T*            sendbuf_,
      std::size_t         sendcount_,
      T*                  recvbuf_,
      std::size_t         recvcount_,
      mpi::Context const& comm_)
    : sendbuf(sendbuf_)
    , sendcount(sendcount_)
    , sendtype(mpi::type_mapper<T>::type())
    , recvbuf(recvbuf_)
    , recvcount(recvcount_)
    , recvtype(mpi::type_mapper<T>::type())
    , comm(comm_) {
    using mapper = mpi::type_mapper<T>;
    static_assert(
        static_cast<bool>(mapper::is_basic),
        "Unknown MPI Type, this probably wouldn't work.");
  }

 private:
  const void* const   sendbuf;
  std::size_t const   sendcount = 0;
  MPI_Datatype const  sendtype  = MPI_DATATYPE_NULL;
  void* const         recvbuf;
  std::size_t const   recvcount = 0;
  MPI_Datatype const  recvtype  = MPI_DATATYPE_NULL;
  mpi::Context const& comm;
};

enum class algorithm
{
  one_factor,
  flat_handshake,
  bruck
};

void run_algorithm(CollectiveArgs args, algorithm algo, std::size_t winsz);
#endif

bool read_input(int argc, char* argv[], struct Params& params);

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim = "\n");

}  // namespace benchmark
#endif
