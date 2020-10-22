#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <chrono>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <fmpi/util/Trace.hpp>
#include <iosfwd>
#include <string>
#include <vector>

namespace benchmark {

namespace detail {
using namespace std::chrono_literals;
static constexpr auto five_seconds = 5s;
}  // namespace detail

struct Params {
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

struct Times {
  using vector_times =
      std::vector<std::pair<std::string, std::chrono::nanoseconds>>;

  vector_times             traces;
  std::chrono::nanoseconds total_time{};
};

struct Measurement {
  size_t nhosts;
  size_t nprocs;
  size_t nthreads;
  int    me;

  size_t step;
  size_t nbytes;
  size_t blocksize;
  size_t iter;

  std::string algorithm;
  Times       times;
};

bool operator<(const Times& lhs, const Times& rhs);

struct CollectiveArgs {
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
        mapper::is_basic, "Unknown MPI Type, this probably wouldn't work.");
  }

  const void* const   sendbuf;
  std::size_t const   sendcount = 0;
  MPI_Datatype const  sendtype  = MPI_DATATYPE_NULL;
  void* const         recvbuf;
  std::size_t const   recvcount = 0;
  MPI_Datatype const  recvtype  = MPI_DATATYPE_NULL;
  mpi::Context const& comm;
};

template <class S, class R>
struct TypedCollectiveArgs : public CollectiveArgs {
  using send_type = S;
  using recv_type = R;
  constexpr TypedCollectiveArgs(
      S const*            sendbuf_,
      std::size_t         sendcount_,
      R*                  recvbuf_,
      std::size_t         recvcount_,
      mpi::Context const& comm_)
    : CollectiveArgs(sendbuf_, sendcount_, recvbuf_, recvcount_, comm_) {
  }
};

bool read_input(int argc, char* argv[], struct Params& params);

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim = "\n");

void write_csv_header(std::ostream& os);

void write_csv(
    std::ostream& os, Measurement const& params, Times const& times);

}  // namespace benchmark
#endif
