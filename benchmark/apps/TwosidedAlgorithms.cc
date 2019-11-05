#include <parallel/algorithm>

#include <tlx/container/simple_vector.hpp>

#include <fmpi/AlltoAll.h>
#include <fmpi/Bruck.h>
#include <fmpi/Math.h>

#include <MPISynchronizedBarrier.h>
#include <rtlx/ScopedLambda.h>

#include <Params.h>
#include <Random.h>
#include <TwosidedAlgorithms.h>

#ifdef NDEBUG
constexpr int nwarmup = 1;
constexpr int niters  = 10;
#else
constexpr int nwarmup = 0;
constexpr int niters  = 1;
#endif

// The container where we store our
using value_t = int;
using storage_t =
    tlx::SimpleVector<value_t, tlx::SimpleVectorMode::NoInitNoDestroy>;
using iterator_t = typename storage_t::iterator;

template <class InputIterator, class OutputIterator>
using merger_t = std::function<void(
    std::vector<std::pair<InputIterator, InputIterator>>, OutputIterator)>;

std::vector<std::pair<
    std::string,
    fmpi_algrithm_t<iterator_t, merger_t<iterator_t, iterator_t>>>>
    ALGORITHMS = {std::make_pair(
                      "AlltoAll",
                      fmpi::MpiAlltoAll<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "ScatteredPairwiseFlatHandshake",
                      fmpi::scatteredPairwise<
                          fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "ScatteredPairwiseOneFactor",
                      fmpi::scatteredPairwise<
                          fmpi::AllToAllAlgorithm::ONE_FACTOR,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeFlatHandshake4",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeFlatHandshake8",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeFlatHandshake16",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::FLAT_HANDSHAKE,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeOneFactor4",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::ONE_FACTOR,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          4>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeOneFactor8",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::ONE_FACTOR,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          8>),
                  std::make_pair(
                      "ScatteredPairwiseWaitsomeOneFactor16",
                      fmpi::scatteredPairwiseWaitsome<
                          fmpi::AllToAllAlgorithm::ONE_FACTOR,
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>,
                          16>),
#if 0
                  std::make_pair(
                      "ScatteredPairwiseWaitany16",
                      fmpi::scatteredPairwiseWaitany<
                          iterator_t,
                          iterator_t,
                          merger_t,
                          16>),
                  std::make_pair(
                      "ScatteredPairwiseWaitany32",
                      fmpi::scatteredPairwiseWaitany<
                          iterator_t,
                          iterator_t,
                          merger_t,
                          32>),
                  std::make_pair(
                      "ScatteredPairwiseWaitany64",
                      fmpi::scatteredPairwiseWaitany<
                          iterator_t,
                          iterator_t,
                          merger_t,
                          64>),
                  std::make_pair(
                      "ScatteredPairwiseWaitany128",
                      fmpi::scatteredPairwiseWaitany<
                          iterator_t,
                          iterator_t,
                          merger_t,
                          128>),
#endif
                  // Bruck Algorithms, first the original one, then a modified
                  // version which omits the last local rotation step
                  std::make_pair(
                      "Bruck",
                      fmpi::bruck<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>)
#if 0
                  std::make_pair(
                      "Bruck_Mod",
                      fmpi::bruck_mod<
                          iterator_t,
                          iterator_t,
                          merger_t<iterator_t, iterator_t>>)
#endif
    };

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::Context worldCtx{MPI_COMM_WORLD};

  auto me = worldCtx.rank();
  auto nr = worldCtx.size();

  fmpi::benchmark::Params params{};

  if (!fmpi::benchmark::process(argc, argv, worldCtx, params)) {
    return -1;
  }

  RTLX_ASSERT((nr % params.nhosts) == 0);

  std::size_t nsteps = std::ceil(std::log2(params.maxblocksize)) -
                       std::ceil(std::log2(params.minblocksize));

  nsteps = std::min(nsteps, std::size_t(20));

  RTLX_ASSERT(params.minblocksize >= sizeof(value_t));

  if (me == 0) {
    printMeasurementHeader(std::cout);
  }

  // calibrate clock
  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(worldCtx.mpiComm());
  RTLX_ASSERT(is_clock_synced);

  FMPI_DBG(params.niters);

  for (size_t blocksize = params.minblocksize, step = 0; step <= nsteps;
       blocksize *= 2, ++step) {
    // each process sends sencount to all other PEs
    auto sendcount = blocksize / sizeof(value_t);

    FMPI_DBG(sendcount);

    RTLX_ASSERT(blocksize % sizeof(value_t) == 0);

    // Required by good old 32-bit MPI
    RTLX_ASSERT(sendcount > 0 && sendcount < std::numeric_limits<int>::max());

    auto nels = sendcount * nr;

    auto data    = storage_t(nels);
    auto out     = storage_t(nels);
    auto correct = storage_t(0);

    for (auto it = 0; it < static_cast<int>(params.niters) + nwarmup; ++it) {
#pragma omp parallel
      {
        std::mt19937_64 generator(random_seed_seq::get_instance());
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

      for (std::size_t block = 0; block < std::size_t(nr); ++block) {
        RTLX_ASSERT(std::is_sorted(
            std::next(data.begin(), block * sendcount),
            std::next(data.begin(), (block + 1) * sendcount)));
      }

      auto merger = [nels](
                        std::vector<std::pair<iterator_t, iterator_t>> seqs,
                        iterator_t                                     res,
                        std::uint16_t nthreads = 0) {
        // parallel merge does not support inplace merging
        // nels must be the number of elements in all sequences
        RTLX_ASSERT(seqs.size());
        RTLX_ASSERT(res);
#if 1
        __gnu_parallel::multiway_merge(
            std::begin(seqs),
            std::end(seqs),
            res,
            nels,
            std::less<>{},
            __gnu_parallel::parallel_tag{nthreads});
#else
        static_cast<void>(nthreads);

        for (auto&& seq : seqs) {
          std::copy(seq.first, seq.second, res);
          res += std::distance(seq.first, seq.second);
        }
#endif
      };

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
      if (params.check) {
        correct = storage_t(nels);
        fmpi::MpiAlltoAll(
            data.begin(),
            correct.begin(),
            static_cast<int>(sendcount),
            worldCtx,
            merger);
      }

      Measurement m{};
      m.nhosts    = params.nhosts;
      m.nprocs    = nr;
      m.me        = me;
      m.step      = step + 1;
      m.nbytes    = nels * nr * sizeof(value_t);
      m.blocksize = blocksize;

      for (auto const& algo : ALGORITHMS) {
        FMPI_DBG_STREAM("running algorithm: " << algo.first);

        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(worldCtx.mpiComm());
        RTLX_ASSERT(barrier.Success(worldCtx.mpiComm()));

        auto t = run_algorithm(
            algo.second,
            data.begin(),
            out.begin(),
            static_cast<int>(sendcount),
            worldCtx,
            merger);

        if (params.check) {
          auto check = std::equal(
              correct.begin(), std::next(correct.begin(), nels), out.begin());

          if (!check) {
            std::ostringstream os;
            os << "[ERROR] [Rank " << me << "] " << algo.first
               << ": incorrect sequence (";
            std::copy(
                out.begin(),
                out.end(),
                std::ostream_iterator<value_t>(os, ", "));
            os << ") vs. (";
            std::copy(
                correct.begin(),
                correct.end(),
                std::ostream_iterator<value_t>(os, ", "));
            os << ")\n";
            std::cout << os.str();
          }
        }

        if (it >= nwarmup) {
          auto trace = rtlx::TimeTrace{me, algo.first};

          m.algorithm = algo.first;

          m.times = std::make_tuple(
              t,
              trace.lookup(fmpi::MERGE),
              trace.lookup(fmpi::COMMUNICATION));

          printMeasurementCsvLine(std::cout, m);

          trace.clear();
        }
      }

      // synchronize before advancing to the next stage
      FMPI_DBG("Iteration Finished");
      MPI_Barrier(worldCtx.mpiComm());
    }
  }

  return 0;
}

void printMeasurementHeader(std::ostream& os)
{
  os << "Nodes, Procs, Round, NBytes, Blocksize, Algo, Rank, Ttotal, "
        "Tmerge, "
        "Tcomm\n";
}

void printMeasurementCsvLine(std::ostream& os, Measurement const& m)
{
  double total, tmerge, tcomm;
  std::tie(total, tmerge, tcomm) = m.times;
  std::ostringstream myos;
  myos << m.nhosts << ", ";
  myos << m.nprocs << ", ";
  myos << m.step << ", ";
  myos << m.nbytes << ", ";
  myos << m.blocksize << ", ";
  myos << m.algorithm << ", ";
  myos << m.me << ", ";
  myos << total << ", ";
  myos << tmerge << ", ";
  myos << tcomm << "\n";
  os << myos.str();
}
