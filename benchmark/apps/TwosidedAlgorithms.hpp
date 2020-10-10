#ifndef TWOSIDEDALGORITHMS_HPP  // NOLINT
#define TWOSIDEDALGORITHMS_HPP  // NOLINT

#include <fmpi/Alltoall.hpp>
//#include <fmpi/Bruck.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/Trace.hpp>
#include <functional>
#include <iostream>
#include <regex>
#include <rtlx/Timer.hpp>
#include <rtlx/UnorderedMap.hpp>

namespace benchmark {

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
};

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

void write_csv_header(std::ostream& os);

void write_csv_line(
    std::ostream&      os,
    Measurement const& params,
    std::pair<
        typename fmpi::TraceStore::key_type,
        typename fmpi::TraceStore::mapped_type> const& entry);

class Runner {
 public:
  template <typename T>
  explicit Runner(const T& obj)
    : object(std::make_shared<Model<T>>(std::move(obj))) {
  }

  [[nodiscard]] std::string_view name() const {
    return object->name();
  }

  template <class S, class R>
  [[nodiscard]] std::chrono::nanoseconds run(
      TypedCollectiveArgs<S, R> args) const {
    using duration = rtlx::steady_timer::duration;

    duration d{};
    {
      rtlx::steady_timer t{d};
      auto               f = object->run(args);
      // automatically waits for future
    }
    return d;
  }

 private:
  struct Concept {
    virtual ~Concept() {
    }
    [[nodiscard]] virtual std::string_view        name() const = 0;
    [[nodiscard]] virtual fmpi::collective_future run(
        CollectiveArgs coll_args) const = 0;
  };

  template <typename T>
  struct Model : Concept {
    explicit Model(T t)
      : object(std::move(t)) {
    }
    [[nodiscard]] std::string_view name() const override {
      return object.name();
    }

    [[nodiscard]] fmpi::collective_future run(
        CollectiveArgs coll_args) const override {
      return object.run(coll_args);
    }

   private:
    T object;
  };
  std::shared_ptr<const Concept> object;
};

std::vector<Runner> algorithm_list(
    std::string const& pattern, mpi::Context const& ctx);

namespace detail {

template <
    class Schedule,
    fmpi::ScheduleOpts::WindowType WinT,
    std::size_t                    Size>
std::string schedule_name() {
  using enum_t = fmpi::ScheduleOpts::WindowType;

  static const std::unordered_map<enum_t, std::string_view> names = {
      {enum_t::sliding, "Waitsome"}, {enum_t::fixed, "Waitall"}};

  return std::string{Schedule::name()} + std::string{names.at(WinT)} +
         std::to_string(Size);
}

}  // namespace detail

template <
    class Schedule,
    fmpi::ScheduleOpts::WindowType WinT,
    std::size_t                    NReqs>
class Alltoall_Runner {
  std::string name_;

 public:
  Alltoall_Runner()
    : name_(detail::schedule_name<Schedule, WinT, NReqs>()) {
  }
  [[nodiscard]] std::string_view name() const noexcept {
    return name_;
  }

  [[nodiscard]] fmpi::collective_future run(CollectiveArgs coll_args) const {
    auto sched = Schedule{coll_args.comm};
    auto opts  = fmpi::ScheduleOpts{sched, NReqs, name(), WinT};
    return fmpi::alltoall(
        coll_args.sendbuf,
        coll_args.sendcount,
        coll_args.sendtype,
        coll_args.recvbuf,
        coll_args.recvcount,
        coll_args.recvtype,
        coll_args.comm,
        opts);
  }
};

template <fmpi::ScheduleOpts::WindowType WinT, std::size_t NReqs>
class Alltoall_Runner<void, WinT, NReqs> {
  static constexpr auto algo_name = std::string_view("AlltoAll");

 public:
  [[nodiscard]] std::string_view name() const noexcept {
    return algo_name;
  }

  [[nodiscard]] fmpi::collective_future run(CollectiveArgs coll_args) const {
    auto request = std::make_unique<MPI_Request>();

    FMPI_CHECK_MPI(MPI_Ialltoall(
        coll_args.sendbuf,
        coll_args.sendcount,
        coll_args.sendtype,
        coll_args.recvbuf,
        coll_args.recvcount,
        coll_args.recvtype,
        coll_args.comm.mpiComm(),
        request.get()));

    return fmpi::make_mpi_future(std::move(request));
  }
};
}  // namespace benchmark

#endif
