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

#include <regex>

#include <fmpi/AlltoAll.hpp>
#include <fmpi/Bruck.hpp>
#include <fmpi/Random.hpp>
#include <fmpi/concurrency/OpenMP.hpp>

#include <rtlx/ScopedLambda.hpp>
#include <rtlx/UnorderedMap.hpp>

#include <tlx/algorithm.hpp>

#include <MPISynchronizedBarrier.hpp>
#include <Params.hpp>
#include <TwosidedAlgorithms.hpp>

// The container where we store our
using value_t = int;
using container =
    tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;

int main(int argc, char* argv[]) {
  // Initialize MPI
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  const auto& world = mpi::Context::world();

  fmpi::benchmark::Params params{};
  if (!fmpi::benchmark::process(argc, argv, world, params)) {
    return 0;
  }

  auto const me = world.rank();
  auto const nr = world.size();

  if ((nr % params.nhosts) != 0) {
    if (me == 0) {
      std::cout << "number of ranks must be equal on all nodes\n";
    }
    return 1;
  }

  using iterator_t = typename container::iterator;
  auto merger      = [](std::vector<std::pair<iterator_t, iterator_t>> seqs,
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

  {
    MPI_Barrier(world.mpiComm());
    std::ostringstream os;

    if (me == 0) {
      os << "Algorithms:\n";

      for (auto&& kv : ALGORITHMS) {
        os << kv.first << "\n";
      }

      os << "\nNode Topology:\n";
    }

    int32_t const ppn = nr / params.nhosts;

    FMPI_DBG(ppn);

    if (me < ppn) {
      os << "  MPI Rank " << me << "\n";
      fmpi::print_pinning(os);
    }

    std::cout << os.str();
    MPI_Barrier(world.mpiComm());
  }

  if (me == 0) {
    fmpi::benchmark::printBenchmarkPreamble(std::cout, "++ ", "\n");
    write_csv_header(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(world.mpiComm());
  assert(is_clock_synced);

  FMPI_DBG(params.niters);

  for (std::size_t step = 0; step < params.sizes.size(); ++step) {
    auto const blocksize = params.sizes[step];
    assert(blocksize >= sizeof(value_t));
    assert(blocksize % sizeof(value_t) == 0);

    auto const sendcount = blocksize / sizeof(value_t);

    FMPI_DBG(sendcount);

    assert(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    assert(sendcount > 0 && sendcount < mpi::max_int);

    auto const nels = sendcount * nr;

    auto data    = container(nels);
    auto out     = container(nels);
    auto correct = container(0);

    for (auto it = 0; it < static_cast<int>(params.niters) + nwarmup; ++it) {
#pragma omp parallel default(none) shared(data) \
    firstprivate(nr, sendcount, me, nels)
      {
        std::random_device r;
        std::seed_seq      seed_seq{r(), r(), r(), r(), r(), r()};
        std::mt19937_64    generator(seed_seq);
        std::uniform_int_distribution<value_t> distribution(-1E6, 1E6);
#pragma omp for
        for (std::size_t block = 0; block < std::size_t(nr); ++block) {
#ifdef NDEBUG
          // generate some random values
          std::generate(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount),
              [&]() { return distribution(generator); });
          // sort it
          std::sort(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount));
#else
          std::iota(
              std::next(data.begin(), block * sendcount),
              std::next(data.begin(), (block + 1) * sendcount),
              block * sendcount + (me * nels));
#endif
        }
      }

      auto& traceStore = fmpi::TraceStore::instance();

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      if (params.check) {
        correct = container(nels);
        fmpi::mpi_alltoall(
            data.begin(),
            correct.begin(),
            static_cast<int>(sendcount),
            world,
            merger);
        traceStore.erase(fmpi::kAlltoall);
        assert(traceStore.empty());
      }

      Measurement m{};
      m.nhosts    = params.nhosts;
      m.nprocs    = nr;
      m.nthreads  = omp_get_max_threads();
      m.me        = me;
      m.step      = step + 1;
      m.nbytes    = nels * nr * sizeof(value_t);
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
            data.begin(),
            out.begin(),
            static_cast<int>(sendcount),
            world,
            merger);

        if (params.check) {
          validate(
              out.begin(), out.end(), correct.begin(), world, algo.first);
        }

        if (it >= nwarmup) {
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
