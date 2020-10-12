#include <MPISynchronizedBarrier.hpp>
#include <Params.hpp>
#include <TwosidedAlgorithms.hpp>
//#include <fmpi/Bruck.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/util/Random.hpp>
#include <regex>
#include <rtlx/ScopedLambda.hpp>

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
  auto const  me    = world.rank();
  auto const  p     = world.size();

  auto const shared_comm = mpi::splitSharedComm(world);
  int const  is_rank0    = static_cast<int>(shared_comm.rank() == 0);
  int        nhosts      = 0;

  int wait = 0;
  while (wait)
    ;

  MPI_Allreduce(
      &is_rank0,
      &nhosts,
      1,
      mpi::type_mapper<int>::type(),
      MPI_SUM,
      world.mpiComm());

  if ((p % nhosts) != 0) {
    if (me == 0) {
      std::cout << "number of ranks must be equal on all nodes\n";
    }
    return 0;
  }

  // using iterator_t = typename container::iterator;

  auto ALGORITHMS = algorithm_list(params.pattern, world);

  if (me == 0) {
    std::ostringstream os;

    os << "Algorithms:\n";

    for (auto&& kv : ALGORITHMS) {
      os << kv.name() << "\n";
    }

    benchmark::printBenchmarkPreamble(os, "++ ", "\n");
    std::cout << os.str() << "\n";
  }

  print_topology(world, nhosts);

  if (me == 0) {
    benchmark::write_csv_header(std::cout);
  }

  // calibrate clock
  auto                  clock           = SynchronizedClock{};
  [[maybe_unused]] bool is_clock_synced = clock.Init(world.mpiComm());
  assert(is_clock_synced);

  if (params.smin < sizeof(value_t)) {
    params.smin = sizeof(value_t);
  }

  params.smax = std::max(params.smin, params.smax);

  for (std::size_t blocksize = params.smin; blocksize <= params.smax;
       blocksize <<= 1) {
    auto const sendcount = blocksize / sizeof(value_t);

    // Required by good old 32-bit MPI
    assert(sendcount < mpi::max_int);

    auto const nels = sendcount * p;

    using vector =
        tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto   sbuf = vector(nels);
    auto   rbuf = vector(nels);
    vector correct;
    // std::size_t step = 1u;

    auto coll_args = benchmark::TypedCollectiveArgs{
        sbuf.data(), sendcount, rbuf.data(), sendcount, world};

    auto const niters = static_cast<int>(params.niters + params.warmups);

    std::unordered_map<std::string, std::vector<benchmark::Times>>
        measurements;

    benchmark::Measurement m{};
    m.nhosts   = nhosts;
    m.nprocs   = p;
    m.nthreads = omp_get_max_threads();
    m.me       = me;
    // m.step      = step++;
    m.nbytes    = nels * p * sizeof(value_t);
    m.blocksize = blocksize;

    for (int it = 0; it < niters; ++it) {
      init_sbuf(sbuf.begin(), sbuf.end(), p, me);

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      if (params.check) {
        correct  = vector(nels);
        auto ret = MPI_Alltoall(
            sbuf.data(),
            sendcount,
            mpi::type_mapper<value_t>::type(),
            correct.data(),
            sendcount,
            mpi::type_mapper<value_t>::type(),
            world.mpiComm());

        assert(ret == MPI_SUCCESS);
        std::sort(std::begin(correct), std::end(correct));
      }

      for (auto&& algo : ALGORITHMS) {
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto       barrier         = clock.Barrier(world.mpiComm());
        auto const barrier_success = barrier.Success(world.mpiComm());
        assert(barrier_success);

        auto times = algo.run(coll_args);

        if (params.warmups == 0 || it > static_cast<int>(params.warmups)) {
          measurements[std::string(algo.name())].push_back(times);
        }

        if (params.check) {
          auto const is_equal =
              std::equal(rbuf.begin(), rbuf.end(), correct.begin());
          if (!is_equal) {
            std::ostringstream os;
            os << "[ERROR] [Rank " << me << "] " << algo.name()
               << ": incorrect sequence";
            MPI_Abort(world.mpiComm(), 1);
            return 1;
          }
        }

        // if (params.warmups == 0 || it > static_cast<int>(params.warmups)) {
        //  m.algorithm = algo.name();
        //  m.iter      = it - params.warmups + 1;

        //  benchmark::write_csv(std::cout, m, times);
        //}
      }
    }

    for (auto&& algo : ALGORITHMS) {
      auto& results = measurements[std::string(algo.name())];
      std::sort(std::begin(results), std::end(results));

      m.algorithm = algo.name();
      auto med    = (results.size() + 1) / 2;

      benchmark::write_csv(std::cout, m, results[med]);
    }
  }

  return 0;
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

template <fmpi::ScheduleOpts::WindowType WinT, std::size_t NReqs>
class Alltoall_Runner<void, WinT, NReqs> {
 public:
  [[nodiscard]] constexpr std::string_view name() const noexcept {
    using namespace std::literals::string_view_literals;
    return "Alltoall"sv;
  }

  [[nodiscard]] fmpi::collective_future run(
      benchmark::CollectiveArgs coll_args) const {
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
