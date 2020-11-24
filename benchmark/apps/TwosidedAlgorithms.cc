#include <MPISynchronizedBarrier.hpp>
#include <Params.hpp>
#include <TwosidedAlgorithms.hpp>
//#include <fmpi/Bruck.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/util/Random.hpp>
#include <regex>
#include <rtlx/ScopedLambda.hpp>
#include <tlx/math/round_to_power_of_two.hpp>

template <class T>
void init_sbuf(T* first, T* last, uint32_t p, int32_t me);

void print_topology(mpi::Context const& ctx, std::size_t nhosts);

int main(int argc, char* argv[]) {
  // Our value type
  using value_t = double;
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  benchmark::Params params{};
  if (!benchmark::read_input(argc, argv, params)) {
    return 0;
  }

  const auto& world = mpi::Context::world();

  auto const nhosts = num_nodes(world);

  if ((world.size() % nhosts) != 0) {
    if (world.rank() == 0) {
      std::cout << "number of ranks must be equal on all nodes\n";
    }
    return 0;
  }

  auto algorithms = algorithm_list(params.pattern, world);

  if (world.rank() == 0) {
    std::ostringstream os;

    os << "Algorithms:\n";

    for (auto&& kv : algorithms) {
      os << kv.name() << "\n";
    }

    benchmark::printBenchmarkPreamble(os, "++ ", "\n");
    std::cout << os.str() << "\n";
  }

  print_topology(world, nhosts);

  if (algorithms.empty()) {
    return EXIT_SUCCESS;
  }

  MPI_Barrier(world.mpiComm());

  if (world.rank() == 0) {
    benchmark::write_csv_header(std::cout);
  }

  // params.smax = tlx::round_down_to_power_of_two(params.smax + 1);
  // params.smin = tlx::round_up_to_power_of_two(params.smin + 1);

  if (params.smin < sizeof(value_t)) {
    params.smin = sizeof(value_t);
  }

  params.smax = std::max(params.smax, params.smin);
  params.pmax = std::max(params.pmax, params.pmin);

  // Array Buffers: do not use make_unique as it has value initialization,
  // which we do not want.
  using storage       = std::unique_ptr<value_t[]>;
  auto const max_size = params.smax * params.pmax / sizeof(value_t);
  auto       sbuf     = storage(new value_t[max_size]);
  auto       rbuf     = storage(new value_t[max_size]);
  auto       correct  = storage(new value_t[(params.check) ? max_size : 0]);

  fmpi::SimpleVector<int> ranks(world.size());
  std::iota(std::begin(ranks), std::end(ranks), 0);

  for (uint32_t nprocs = params.pmin; nprocs < params.pmax + 1; nprocs *= 2) {
    MPI_Barrier(world.mpiComm());

    /* create communicator for nprocs */
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Comm  comm  = MPI_COMM_NULL;
    MPI_Group_incl(world.mpiGroup(), nprocs, ranks.data(), &group);
    MPI_Comm_create(world.mpiComm(), group, &comm);
    MPI_Group_free(&group);

    if (comm == MPI_COMM_NULL) {
      continue;
    }

    mpi::Context ctx{comm};
    FMPI_DBG(ctx.size());

    assert(ctx.size() == nprocs);

    auto                  clock           = SynchronizedClock{};
    [[maybe_unused]] bool is_clock_synced = clock.Init(ctx.mpiComm());
    assert(is_clock_synced);

    for (std::size_t blocksize = params.smin; blocksize <= params.smax;
         blocksize <<= 1) {
      auto const sendcount = blocksize / sizeof(value_t);

      // Required by good old 32-bit MPI
      assert(sendcount < mpi::max_int);

      auto const nels = sendcount * ctx.size();

      assert(nels <= max_size);

      auto coll_args = benchmark::TypedCollectiveArgs{
          sbuf.get(), sendcount, rbuf.get(), sendcount, ctx};

      for (auto&& benchmark : algorithms) {
        for (auto&& w : fmpi::range(0u, params.warmups)) {
          std::ignore = w;
          // warumup iterations
          init_sbuf(sbuf.get(), sbuf.get() + nels, ctx.size(), ctx.rank());

          if (params.check) {
            calculate_correct_result(benchmark::TypedCollectiveArgs{
                sbuf.get(), sendcount, correct.get(), sendcount, ctx});
          }

          std::ignore = benchmark.run(coll_args);

          if (params.check and
              not check_result(coll_args, correct.get(), benchmark.name())) {
            return 1;
          }
        }

        MPI_Barrier(ctx.mpiComm());

        fmpi::SimpleVector<benchmark::Times> results(params.niters);

        for (auto&& i : fmpi::range(params.niters)) {
          init_sbuf(sbuf.get(), sbuf.get() + nels, ctx.size(), ctx.rank());
          // We always want to guarantee that all processors start at the same
          // time, so this is a real barrier
          auto       barrier         = clock.Barrier(ctx.mpiComm());
          auto const barrier_success = barrier.Success(ctx.mpiComm());
          assert(barrier_success);

          auto times = benchmark.run(coll_args);

          results[i] = times;
        }

        if (results.empty()) {
          continue;
        }

        // median of measurements
        auto const middle =
            std::min(results.size() - 1, (results.size() + 1) / 2);

        auto const med = std::next(std::begin(results), middle);
        std::sort(std::begin(results), std::end(results));

        benchmark::Measurement m{};
        m.nhosts    = num_nodes(ctx);
        m.nprocs    = ctx.size();
        m.nthreads  = omp_get_max_threads();
        m.me        = ctx.rank();
        m.nbytes    = nels * sizeof(value_t);
        m.blocksize = blocksize;
        m.algorithm = benchmark.name();

        benchmark::write_csv(std::cout, m, *med);
        MPI_Barrier(ctx.mpiComm());
      }
    }
  }

  return EXIT_SUCCESS;
}

template <class T>
void init_sbuf(T* first, T* last, uint32_t p, int32_t me) {
  auto const nels = std::distance(first, last);
  assert(nels > 0);

  auto const sendcount = nels / p;

#pragma omp parallel default(none) shared(first, last) \
    firstprivate(p, sendcount, me, nels)
  {
#ifdef NDEBUG
    using uniform_dist = std::conditional_t<
        std::is_integral_v<T>,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>>;
    std::random_device r;
    std::seed_seq      seed_seq{r(), r(), r(), r(), r(), r()};
    std::mt19937_64    generator(seed_seq);
    uniform_dist       distribution;
#endif
#pragma omp for
    for (std::size_t block = 0; block < std::size_t(p); ++block) {
#ifdef NDEBUG
      // generate some random values
      std::generate(
          std::next(first, block * sendcount),
          std::next(first, (block + 1) * sendcount),
          [&]() { return distribution(generator); });
      // sort it
      std::sort(
          std::next(first, block * sendcount),
          std::next(first, (block + 1) * sendcount));
#else
      std::iota(
          std::next(first, block * sendcount),
          std::next(first, (block + 1) * sendcount),
          block * sendcount + (me * nels));
#endif
    }
  }
}

void print_topology(mpi::Context const& ctx, std::size_t nhosts) {
  auto const me   = ctx.rank();
  auto const ppn  = static_cast<int32_t>(ctx.size() / nhosts);
  auto const last = mpi::Rank{ppn - 1};

  auto left  = (me > 0 && me <= last) ? me - 1 : mpi::Rank::null();
  auto right = (me < last) ? me + 1 : mpi::Rank::null();

  if (not(left or right)) {
    return;
  }

  char dummy = 0;

  std::ostringstream os;

  if (me == 0) {
    os << "Node Topology:\n";
  }

  MPI_Recv(&dummy, 1, MPI_CHAR, left, 0xAB, ctx.mpiComm(), MPI_STATUS_IGNORE);

  if (me < ppn) {
    os << "  MPI Rank " << me << "\n";
    fmpi::print_pinning(os);
  }

  if (me == last) {
    os << "\n";
  }

  std::cout << os.str() << std::endl;

  MPI_Send(&dummy, 1, MPI_CHAR, right, 0xAB, ctx.mpiComm());

  if (me == 0) {
    MPI_Recv(
        &dummy, 1, MPI_CHAR, last, 0xAB, ctx.mpiComm(), MPI_STATUS_IGNORE);
  } else if (me == last) {
    MPI_Send(&dummy, 1, MPI_CHAR, 0, 0xAB, ctx.mpiComm());
  }
}

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

template <class Schedule, fmpi::ScheduleOpts::WindowType WinT, uint32_t NReqs>
class Alltoall_Runner {
  std::string name_;

 public:
  Alltoall_Runner()
    : name_(detail::schedule_name<Schedule, WinT, NReqs>()) {
  }
  [[nodiscard]] std::string_view name() const noexcept {
    return name_;
  }

  [[nodiscard]] fmpi::collective_future run(
      benchmark::CollectiveArgs coll_args) const {
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

template <fmpi::ScheduleOpts::WindowType WinT, uint32_t NReqs>
class Alltoall_Runner<fmpi::Linear, WinT, NReqs> {
  std::string name_;

 public:
  [[nodiscard]] std::string_view name() const noexcept {
    return std::string_view("Linear");
  }

  [[nodiscard]] fmpi::collective_future run(
      benchmark::CollectiveArgs coll_args) const {
    auto sched = fmpi::Linear{coll_args.comm};
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

template <fmpi::ScheduleOpts::WindowType WinT, uint32_t NReqs>
class Alltoall_Runner<void, WinT, NReqs> {
 public:
  [[nodiscard]] constexpr std::string_view name() const noexcept {
    using namespace std::literals::string_view_literals;
    return "Alltoall"sv;
  }

  [[nodiscard]] fmpi::collective_future run(
      benchmark::CollectiveArgs coll_args) const {
    auto request = fmpi::make_mpi_future();

    FMPI_CHECK_MPI(MPI_Ialltoall(
        coll_args.sendbuf,
        static_cast<int>(coll_args.sendcount),
        coll_args.sendtype,
        coll_args.recvbuf,
        static_cast<int>(coll_args.recvcount),
        coll_args.recvtype,
        coll_args.comm.mpiComm(),
        &request.native_handles().front()));

    return request;
  }
};

std::vector<Runner> algorithm_list(
    std::string const& pattern, mpi::Context const& ctx) {
  using win_t     = fmpi::ScheduleOpts::WindowType;
  auto algorithms = std::vector<Runner>({
    Runner{Alltoall_Runner<void, win_t::fixed, 0>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::fixed, 4>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::fixed, 8>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::fixed, 16>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::sliding, 4>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::sliding, 8>()},
        Runner{Alltoall_Runner<fmpi::FlatHandshake, win_t::sliding, 16>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::fixed, 4>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::fixed, 8>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::fixed, 16>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::sliding, 4>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::sliding, 8>()},
        Runner{Alltoall_Runner<fmpi::OneFactor, win_t::sliding, 16>()},
#if 0
          // Bruck Algorithms, first the original one, then a modified
          // version which omits the last local rotation step
          std::make_pair(
              "Bruck",
              fmpi::bruck<
                  RandomAccessIterator1,
                  RandomAccessIterator2>),
          std::make_pair(
              "Bruck_indexed",
              fmpi::bruck_indexed<
                  RandomAccessIterator1,
                  RandomAccessIterator2>),
          std::make_pair(
              "Bruck_interleave",
              fmpi::bruck_interleave<
                  RandomAccessIterator1,
                  RandomAccessIterator2>),
          std::make_pair(
              "Bruck_interleave_dispatch",
              fmpi::bruck_interleave_dispatch<
                  RandomAccessIterator1,
                  RandomAccessIterator2>),
          std::make_pair(
              "Bruck_Mod",
              fmpi::bruck_mod<
                  RandomAccessIterator1,
                  RandomAccessIterator2>)
#endif
  });

  if (!pattern.empty()) {
    // remove algorithms not matching a pattern
    auto const regex = std::regex(pattern);

    algorithms.erase(
        std::remove_if(
            std::begin(algorithms),
            std::end(algorithms),
            [regex](auto const& entry) {
              std::match_results<std::string_view::const_iterator> base_match;
              auto const& name = entry.name();
              return !std::regex_match(
                  name.begin(), name.end(), base_match, regex);
            }),
        algorithms.end());
  }

  if (!fmpi::isPow2(ctx.size())) {
    algorithms.erase(
        std::remove_if(
            std::begin(algorithms),
            std::end(algorithms),
            [](auto const& entry) {
              return entry.name().find("Bruck_Mod") not_eq
                     std::string_view::npos;
            }),
        algorithms.end());
  }

  FMPI_DBG(algorithms.size());

  return algorithms;
}

uint32_t num_nodes(mpi::Context const& comm) {
  auto const shared_comm = mpi::splitSharedComm(comm);
  int const  is_rank0    = static_cast<int>(shared_comm.rank() == 0);
  int        nhosts      = 0;

  MPI_Allreduce(
      &is_rank0,
      &nhosts,
      1,
      mpi::type_mapper<int>::type(),
      MPI_SUM,
      comm.mpiComm());

  return nhosts;
}

#if 0
  auto merger = [](std::vector<std::pair<iterator_t, iterator_t>> seqs,
                   iterator_t                                     res) {
    // parallel merge does not support inplace merging
    // nels must be the number of elements in all sequences
    assert(!seqs.empty());
    assert(res);

    auto const size = std::accumulate(
        std::begin(seqs),
        std::end(seqs),
        std::size_t(0),
        [](auto acc, auto c) {
          return acc + std::distance(c.first, c.second);
        });

    return tlx::parallel_multiway_merge(
        std::begin(seqs),
        std::end(seqs),
        res,
        size,
        std::less<>{},
        tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
        tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
        omp_get_max_threads());
  };
#endif
