#ifndef TWOSIDEDALGORITHMS_HPP  // NOLINT
#define TWOSIDEDALGORITHMS_HPP  // NOLINT

#include <Merge.hpp>
#include <Params.hpp>
#include <fmpi/Alltoall.hpp>
#include <fmpi/util/Trace.hpp>
//#include <fmpi/Bruck.hpp>
#include <iostream>
#include <regex>
#include <rtlx/Timer.hpp>

static constexpr auto Ttotal = std::string_view("Ttotal");

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
  [[nodiscard]] benchmark::Times run(
      benchmark::TypedCollectiveArgs<S, R> args) const {
    using namespace std::literals::string_view_literals;
    using duration = rtlx::steady_timer::duration;

    benchmark::Times::vector_times times;
    auto                           d_total = duration{};

    using simple_vector =
        tlx::SimpleVector<R, tlx::SimpleVectorMode::NoInitNoDestroy>;

    simple_vector buffer(args.recvcount * args.comm.size());

    {
      rtlx::steady_timer timer{d_total};
      // 1) Communication
      auto future = object->run(args);

      // 2) Computation + Communication Overlap
      times = benchmark::merge_async(args, std::move(future), buffer.data());

      std::move(
          std::begin(buffer),
          std::end(buffer),
          static_cast<R*>(args.recvbuf));

    }  // 3 Stop timer

    times.emplace_back(Ttotal, d_total);

    {
      // Collect Traces
      auto&       traceStore = fmpi::TraceStore::instance();
      auto const& traces     = traceStore.traces(name());

      std::copy(
          std::begin(traces), std::end(traces), std::back_inserter(times));

      traceStore.erase(name());

      assert(traceStore.empty());
    }

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
      : object(std::move(t)) {
    }
    [[nodiscard]] std::string_view name() const override {
      return object.name();
    }

    [[nodiscard]] fmpi::collective_future run(
        benchmark::CollectiveArgs coll_args) const override {
      return object.run(coll_args);
    }

   private:
    T object;
  };
  std::shared_ptr<const Concept> object;
};

std::vector<Runner> algorithm_list(
    std::string const& pattern, mpi::Context const& ctx);

#endif
