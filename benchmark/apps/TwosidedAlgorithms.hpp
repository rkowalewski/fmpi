#ifndef TWOSIDEDALGORITHMS_HPP  // NOLINT
#define TWOSIDEDALGORITHMS_HPP  // NOLINT

#include <Params.hpp>
#include <algorithms/Merge.hpp>
#include <fmpi/Alltoall.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/util/Trace.hpp>
//#include <fmpi/Bruck.hpp>
#include <iostream>
#include <regex>
#include <rtlx/Timer.hpp>

static constexpr auto Ttotal    = std::string_view("Ttotal");
static constexpr auto Tschedule = std::string_view("Tcomm.schedule");

uint32_t num_nodes(mpi::Context const& comm);

class Runner {
 public:
  template <typename T>
  explicit Runner(const T& obj)
    : communication(std::make_shared<Model<T>>(std::move(obj))) {
  }

  [[nodiscard]] std::string_view name() const {
    return communication->name();
  }

  template <class S, class R>
  [[nodiscard]] benchmark::Times run(
      benchmark::TypedCollectiveArgs<S, R> args) const {
    using namespace std::literals::string_view_literals;
    using duration = rtlx::steady_timer::duration;
    using namespace std::chrono_literals;

    benchmark::Times::vector_times times;
    auto                           d_total    = duration{};
    auto                           d_schedule = duration{};

    fmpi::SimpleVector<R> buffer(args.recvcount * args.comm.size());

    {
      rtlx::steady_timer timer{d_total};

      // 1) Communication Schedule
      rtlx::steady_timer t_schedule{d_schedule};
      auto               future = communication->run(args);
      t_schedule.finish();

      // 2) Computation + Communication Overlap
      times = benchmark::merge_async(args, std::move(future), buffer.data());

      std::move(
          std::begin(buffer),
          std::end(buffer),
          static_cast<R*>(args.recvbuf));

    }  // 3 Stop timer

    times.emplace_back(Ttotal, d_total);
    times.emplace_back(Tschedule, d_schedule);

#if 0
    {
      // Collect Traces
      auto&       traceStore = fmpi::TraceStore::instance();
      auto const& traces     = traceStore.traces(name());

      std::copy(
         std::begin(traces), std::end(traces), std::back_inserter(times));

      traceStore.erase(name());

      assert(traceStore.empty());
    }
#endif

    auto comms = std::partition(
        std::begin(times), std::end(times), [](const auto& pair) {
          return pair.first.find("Tcomm") not_eq std::string::npos;
        });

    auto t_comm = std::accumulate(
        std::begin(times), comms, 0ns, [](const auto& acc, const auto& pair) {
          return acc + pair.second;
        });

    times.emplace_back("Tcomm", t_comm);

    return benchmark::Times{times, d_total};
  }

 private:
  struct Concept {
    virtual ~Concept() {
    }
    [[nodiscard]] virtual std::string_view        name() const = 0;
    [[nodiscard]] virtual fmpi::collective_future run(
        benchmark::CollectiveArgs coll_args) const = 0;
  };

  template <typename T>
  struct Model : Concept {
    explicit Model(T t)
      : communication(std::move(t)) {
    }
    [[nodiscard]] std::string_view name() const override {
      return communication.name();
    }

    [[nodiscard]] fmpi::collective_future run(
        benchmark::CollectiveArgs coll_args) const override {
      return communication.run(coll_args);
    }

   private:
    T communication;
  };
  std::shared_ptr<const Concept> communication;
};

std::vector<Runner> algorithm_list(
    std::string const& pattern, mpi::Context const& ctx);

template <class S, class R>
void calculate_correct_result(benchmark::TypedCollectiveArgs<S, R> args) {
  fmpi::SimpleVector<R> buffer(args.recvcount * args.comm.size());

  auto ret = MPI_Alltoall(
      args.sendbuf,
      static_cast<int>(args.sendcount),
      args.sendtype,
      args.recvbuf,
      static_cast<int>(args.recvcount),
      args.recvtype,
      args.comm.mpiComm());

  auto future = fmpi::make_ready_future(ret);

  benchmark::merge_async(args, std::move(future), buffer.data());

  std::move(
      std::begin(buffer), std::end(buffer), static_cast<R*>(args.recvbuf));
}

template <class S, class R>
bool check_result(
    benchmark::TypedCollectiveArgs<S, R> args,
    R const*                             correct_begin,
    std::string_view                     name) {
  auto const* recvbuf = static_cast<R*>(args.recvbuf);
  auto const  size    = args.recvcount * args.comm.size();

  auto const is_equal = std::equal(recvbuf, recvbuf + size, correct_begin);

  if (!is_equal) {
    std::ostringstream os;
    os << "[ERROR] [Rank " << args.comm.rank() << "] " << name
       << ": incorrect sequence -- ";
    //    std::copy(recvbuf, recvbuf + size, std::ostream_iterator<R>(os, ",
    //    ")); os << "\ncorrect: ";
    //
    //    std::copy(
    //        correct_begin,
    //        correct_begin + size,
    //        std::ostream_iterator<R>(os, ", "));
    //
    os << "\n";

    std::cout << os.str();
    MPI_Abort(args.comm.mpiComm(), 1);
  }

  return is_equal;
}

#endif
