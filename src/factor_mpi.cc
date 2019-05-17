#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include <Benchmark.h>
#include <Bruck.h>
#include <Debug.h>
#include <Random.h>
#include <Timer.h>
#include <Types.h>

#include <synchronized_barrier.hpp>

constexpr size_t KB = 1 << 10;
constexpr size_t MB = 1 << 20;

constexpr size_t niters       = 10;
constexpr size_t minblocksize = KB;

// This are approximately 25 GB
// constexpr size_t capacity_per_node = 32 * MB * 28 * 28;
constexpr size_t capacity_per_node = MB;

double time_local_rotate;
double time_sendbuf;
double time_recvbuf;
double time_comm;

// The container where we store our
using value_t     = int;
using container_t = std::unique_ptr<value_t[]>;
using iterator_t  = typename container_t::pointer;

using benchmark_t =
    std::function<void(iterator_t, iterator_t, int, MPI_Comm)>;

std::array<std::pair<std::string, benchmark_t>, 6> algos = {
    std::make_pair("AlltoAll", MpiAlltoAll<iterator_t, iterator_t>),
    std::make_pair("FactorParty", factorParty<iterator_t, iterator_t>),
    std::make_pair("FlatFactor", flatFactor<iterator_t, iterator_t>),
    std::make_pair("FlatHandshake", flatHandshake<iterator_t, iterator_t>),
    std::make_pair("Bruck", alltoall_bruck<iterator_t, iterator_t>),
    std::make_pair("Bruck_Mod", alltoall_bruck_mod<iterator_t, iterator_t>)};

int main(int argc, char* argv[])
{
  using measurements_t = std::unordered_map<std::string, std::vector<double>>;

  int         me, nr;
  container_t data, out, correct;

  measurements_t measurements;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nr);

  if (argc == 1) {
    if (me == 0) {
      std::cout << "usage: " << argv[0] << " [number of nodes]\n";
    }
    MPI_Finalize();
    return 1;
  }

  ASSERT(nr >= 1);

  auto nhosts = std::atoi(argv[1]);

  if (me == 0) {
    print_env();
  }

  std::mt19937_64 generator(random_seed_seq::get_instance());

  // We have to half the capacity because we do not in-place all to all
  // We again half by the number of processors
  // const size_t number_nodes = nr / 28;
  ASSERT((nr % nhosts) == 0);

  auto procs_per_node = nr / nhosts;

  auto clock           = SynchronizedClock{};
  bool is_clock_synced = clock.Init(comm);
  ASSERT(is_clock_synced);

  // We divide by two because we have in and out buffers
  // Then we divide by the number of PEs per node
  // Then we divide again divide by number of PEs to obtain the largest
  // blocksize.
  const size_t maxblocksize =
      capacity_per_node / (2 * procs_per_node * procs_per_node);

  for (size_t blocksize = minblocksize, step = 0; blocksize <= maxblocksize;
       blocksize *= 2, ++step) {
    // Each PE sends nchunks to all other PEs
    auto nchunks = blocksize / (sizeof(value_t) * nhosts);

    // Required by good old 32-bit MPI
    ASSERT(nchunks > 0 && nchunks < std::numeric_limits<int>::max());

    auto nels = size_t(nr) * nchunks;

    data.reset(new value_t[nels]);
    out.reset(new value_t[nels]);
#ifndef NDEBUG
    correct.reset(new value_t[nels]);
#endif

    std::iota(&(data[0]), &(data[nels]), me * nr);

    for (size_t it = 0; it < niters; ++it) {
#ifdef NDEBUG
      std::shuffle(std::begin(data), std::end(data), generator);
#endif

      // first we want to obtain the correct result which we can verify then
      // with our own algorithms
#ifndef NDEBUG
      constexpr auto mpi_datatype = mpi::mpi_datatype<
          typename std::iterator_traits<iterator_t>::value_type>::value;
      auto res = MPI_Alltoall(
          &(data[0]),
          nchunks,
          mpi_datatype,
          &(correct[0]),
          nchunks,
          mpi_datatype,
          MPI_COMM_WORLD);
#else
      auto res = MPI_SUCCESS;
#endif

      ASSERT(res == MPI_SUCCESS);

      time_local_rotate = 0;
      time_sendbuf      = 0;
      time_recvbuf      = 0;
      time_comm         = 0;

      for (auto const& algo : algos) {
        // We always want to guarantee that all processors start at the same
        // time, so this is a real barrier
        auto barrier = clock.Barrier(comm);
        ASSERT(barrier.Success(comm));

        auto t = run_algorithm(
            algo.second, &(data[0]), &(out[0]), nchunks, MPI_COMM_WORLD);

        measurements[algo.first].emplace_back(t);

        ASSERT(std::equal(&(correct[0]), &(correct[nels]), &(out[0])));
      }
    }

    std::vector<StringDoublePair> ranking;

    constexpr int root = 0;

    for (auto const& algo : algos) {
      auto mid = (niters / 2);

      auto& results = measurements[algo.first];

      // local median
      std::nth_element(results.begin(), results.begin() + mid, results.end());

      // global median
      auto med = medianReduce(results[mid], root, comm);

      // collect the global median into a vector
      if (me == root) {
        ranking.emplace_back(StringDoublePair{algo.first, med});
      }
    }

    auto medRotate  = medianReduce(time_local_rotate, root, comm);
    auto medSendBuf = medianReduce(time_sendbuf, root, comm);
    auto medRecvBuf = medianReduce(time_recvbuf, root, comm);

    if (me == root) {
      ASSERT(ranking.size() == algos.size());
      // sort the median vector
      std::sort(ranking.begin(), ranking.end());

      std::cout << "(" << step << ") Global Volume (KB) "
                << nels * nr * sizeof(value_t) / KB
                << ", blocksize (KB) = " << nchunks * sizeof(value_t) / KB
                << ", ranking: ";
      // print until second to last
      std::copy(
          std::begin(ranking),
          std::end(ranking) - 1,
          std::ostream_iterator<StringDoublePair>(std::cout, ", "));
      // print last
      std::cout << *(std::prev(ranking.end()));

      std::cout << "\n";

      std::cout << "time rotate  : " << medRotate << "\n"
                << "time send_buf: " << medSendBuf << "\n"
                << "time recv_buf: " << medRecvBuf << "\n";

      // flush stdio buffer
      std::cout << std::endl;
    }

    // reset measurements for next iteration
    measurements.clear();
  }

  MPI_Finalize();

  return 0;
}
