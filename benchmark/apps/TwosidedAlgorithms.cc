/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <omp.h>

#include <MPISynchronizedBarrier.hpp>
#include <Params.hpp>
#include <TwosidedAlgorithms.hpp>
#include <fmpi/AlltoAll.hpp>
#include <fmpi/Bruck.hpp>
#include <fmpi/Random.hpp>
#include <fmpi/concurrency/OpenMP.hpp>
#include <regex>
#include <rtlx/ScopedLambda.hpp>
#include <rtlx/UnorderedMap.hpp>
#include <tlx/algorithm.hpp>

// The container where we store our data
using value_t = double;
using container =
    tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

template <class T>
void init_sbuf(T* first, T* last, uint32_t p, int32_t me);

void print_topology(
    mpi::Context const& ctx, fmpi::benchmark::Params const& params);

int main(int argc, char* argv[]) {
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  fmpi::benchmark::Params params{};
  if (!fmpi::benchmark::read_input(argc, argv, params)) {
    return 0;
  }

  const auto& world = mpi::Context::world();
  auto const  me    = world.rank();
  auto const  p     = world.size();

  if ((p % params.nhosts) != 0) {
    if (me == 0) {
      std::cout << "number of ranks must be equal on all nodes\n";
    }
    return 0;
  }

  using iterator_t = typename container::iterator;

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

  auto ALGORITHMS = algorithm_list<iterator_t, iterator_t, decltype(merger)>(
      params.pattern, world);

  if (me == 0) {
    std::ostringstream os;

    os << "Algorithms:\n";

    for (auto&& kv : ALGORITHMS) {
      os << kv.first << "\n";
    }

    fmpi::benchmark::printBenchmarkPreamble(os, "++ ", "\n");
    std::cout << os.str() << "\n";
  }

  print_topology(world, params);

  if (me == 0) {
    write_csv_header(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(world.mpiComm());
  assert(is_clock_synced);

  FMPI_DBG(params.niters);

  if (params.smin < sizeof(value_t)) {
    params.smin = sizeof(value_t);
  }

  for (std::size_t blocksize = params.smin; blocksize <= params.smax;
       blocksize <<= 1) {
    auto const sendcount = blocksize / sizeof(value_t);
    FMPI_DBG(sendcount);

    // Required by good old 32-bit MPI
    assert(sendcount < mpi::max_int);

    auto const nels = sendcount * p;

    auto        sbuf    = container(nels);
    auto        rbuf    = container(nels);
    auto        correct = container(0);
    std::size_t step    = 1u;

#ifdef NDEBUG
    auto const nwarmup = 1;
#else
    auto const nwarmup = 0;
#endif

    for (auto it = 0; it < static_cast<int>(params.niters) + nwarmup; ++it) {
      auto& traceStore = fmpi::TraceStore::instance();

      init_sbuf(sbuf.begin(), sbuf.end(), p, me);

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      if (params.check) {
        correct = container(nels);
        fmpi::mpi_alltoall(
            sbuf.begin(),
            correct.begin(),
            static_cast<int>(sendcount),
            world,
            merger);
        traceStore.erase(fmpi::kAlltoall);
        assert(traceStore.empty());
      }

      Measurement m{};
      m.nhosts    = params.nhosts;
      m.nprocs    = p;
      m.nthreads  = omp_get_max_threads();
      m.me        = me;
      m.step      = step++;
      m.nbytes    = nels * p * sizeof(value_t);
      m.blocksize = blocksize;

      for (auto&& algo : ALGORITHMS) {
        FMPI_DBG_STREAM("running algorithm: " << algo.first);

        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto       barrier         = clock.Barrier(world.mpiComm());
        auto const barrier_success = barrier.Success(world.mpiComm());
        assert(barrier_success);

        auto total = run_algorithm(

            algo.second,
            sbuf.begin(),
            rbuf.begin(),
            static_cast<int>(sendcount),
            world,
            merger);

        if (params.check) {
          validate(
              rbuf.begin(), rbuf.end(), correct.begin(), world, algo.first);
        }

        if (nwarmup == 0 || it > nwarmup) {
          m.algorithm = algo.first;
          m.iter      = it - nwarmup + 1;

          auto const& traces = traceStore.traces(algo.first);

          write_csv_line(
              std::cout,
              m,
              std::make_pair(std::string{fmpi::kTotalTime}, total));

          for (auto&& entry : traces) {
            write_csv_line(std::cout, m, entry);
          }
        }

        traceStore.erase(algo.first);
      }
      assert(traceStore.empty());
    }
  }

  return 0;
}

template <class Rep, class Period>
std::ostream& operator<<(
    std::ostream& os, const std::chrono::duration<Rep, Period>& d) {
  os << rtlx::to_seconds(d);
  return os;
}

void write_csv_header(std::ostream& os) {
  os << "Nodes, Procs, Threads, Round, NBytes, Blocksize, Algo, Rank, "
        "Iteration, "
        "Measurement, "
        "Value\n";
}

std::ostream& operator<<(
    std::ostream& os, typename fmpi::TraceStore::mapped_type const& v) {
  std::visit([&os](auto const& val) { os << val; }, v);
  return os;
}

void write_csv_line(
    std::ostream&      os,
    Measurement const& params,
    std::pair<
        typename fmpi::TraceStore::key_type,
        typename fmpi::TraceStore::mapped_type> const& entry) {
  std::ostringstream myos;
  myos << params.nhosts << ", ";
  myos << params.nprocs << ", ";
  myos << params.nthreads << ", ";
  myos << params.step << ", ";
  myos << params.nbytes << ", ";
  myos << params.blocksize << ", ";
  myos << params.algorithm << ", ";
  myos << params.me << ", ";
  myos << params.iter << ", ";
  myos << entry.first << ", ";
  myos << entry.second << "\n";
  os << myos.str();
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
    std::random_device r;
    std::seed_seq      seed_seq{r(), r(), r(), r(), r(), r()};
    std::mt19937_64    generator(seed_seq);
    std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
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

void print_topology(
    mpi::Context const& ctx, fmpi::benchmark::Params const& params) {
  auto const me   = ctx.rank();
  auto const ppn  = static_cast<int32_t>(ctx.size() / params.nhosts);
  auto const last = mpi::Rank{ppn - 1};

  auto left  = (me > 0 && me <= last) ? me - 1 : mpi::Rank::null();
  auto right = (me < last) ? me + 1 : mpi::Rank::null();

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
