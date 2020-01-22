#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <boost/container/small_vector.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/allocator/ContiguousPoolAllocator.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/detail/Capture.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Dispatcher.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/Request.hpp>
#include <future>
#include <iostream>
#include <numeric>
#include <random>
#include <rtlx/Assert.hpp>
#include <rtlx/ScopedLambda.hpp>
#include <sstream>
#include <thread>

const unsigned long QUEUE_SIZE     = 5L;
const unsigned long TOTAL_ELEMENTS = QUEUE_SIZE * 10L;

std::vector<int> gen_random_vec() {
  auto randomNumberBetween = [](int low, int high) {
    auto randomFunc =
        [distribution_  = std::uniform_int_distribution<int>(low, high),
         random_engine_ = std::mt19937{std::random_device{}()}]() mutable {
          return distribution_(random_engine_);
        };
    return randomFunc;
  };

  std::vector<int> expect;
  std::generate_n(
      std::back_inserter(expect),
      TOTAL_ELEMENTS,
      randomNumberBetween(1, 1000));

  return expect;
}

int test();

int main(int argc, char* argv[]) {
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  if (provided < MPI_THREAD_SERIALIZED) {
    std::cerr << "MPI_THREAD_SERIALIZED not supported\n";
    return 1;
  }

  test();

  return 0;
}

int test() {
  mpi::Context const world{MPI_COMM_WORLD};

  // int one = 1;
  // int buf = 100;

  constexpr std::size_t winsz = 4;
  fmpi::CommDispatcher  dispatcher{winsz};

  constexpr std::size_t blocksize = 1;

  std::vector<int> sbuf(world.size() * blocksize);
  std::vector<int> rbuf(world.size() * blocksize);

  int const first = world.rank() * world.size();
  std::iota(std::begin(sbuf), std::end(sbuf), first);

  std::vector<int> expect(world.size() * blocksize);

  int val = world.rank();
  std::generate(
      std::begin(expect),
      std::end(expect),
      [&val, size = world.size()]() mutable {
        return std::exchange(val, val + size);
      });

  FMPI_DBG_RANGE(std::begin(sbuf), std::end(sbuf));

  dispatcher.start_worker();
  dispatcher.pinToCore(1);

  // sleep for one second
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // FMPI_DBG(&buf);

  constexpr std::size_t n_pipelines = 2;
  constexpr int         mpi_tag     = 0;

  // Pipeline 2
  using rank_data_pair = std::pair<mpi::Rank, fmpi::Span<int>>;

  auto ready_tasks = fmpi::buffered_channel<rank_data_pair>{world.size()};

  uint16_t const size =
      std::min<uint16_t>(winsz, world.size()) * blocksize * 2;

  constexpr bool thread_safe = true;
  auto           buf_alloc   = fmpi::HeapAllocator<int, thread_safe>{size};

  using token_data_pair = std::pair<fmpi::Ticket, fmpi::Span<int>>;
  boost::container::small_vector<token_data_pair, winsz * n_pipelines> blocks;

  for (auto&& p : fmpi::range(world.size())) {
    auto       rb      = fmpi::Span<int>(&rbuf[p], blocksize);
    auto const peer    = static_cast<mpi::Rank>(p);
    auto       rticket = dispatcher.postAsync(
        fmpi::request_type::IRECV,
        [&blocks, &buf_alloc, peer, &world](
            MPI_Request* req, fmpi::Ticket ticket) -> int {
          // allocator some buffer
          auto b = buf_alloc.allocate(blocksize);
          blocks.push_back(
              std::make_pair(ticket, fmpi::make_span(b, blocksize)));
          return mpi::irecv(b, blocksize, peer, mpi_tag, world, req);
        },
        [&blocks, &ready_tasks, peer](
            MPI_Status status, fmpi::Ticket ticket) {
          FMPI_CHECK(status.MPI_ERROR == MPI_SUCCESS);

          auto it = std::find_if(
              std::begin(blocks), std::end(blocks), [ticket](const auto& v) {
                return v.first == ticket;
              });

          FMPI_DBG_RANGE(it->second.begin(), it->second.end());

          ready_tasks.push(std::make_pair(peer, it->second));
          blocks.erase(it);
        });

    auto sb = fmpi::Span<const int>(&sbuf[peer], blocksize);

    auto sticket = dispatcher.postAsync(
        fmpi::request_type::ISEND,
        [sb, peer, &world](MPI_Request* req, fmpi::Ticket) -> int {
          return mpi::isend(
              sb.data(),
              sb.size(),
              static_cast<mpi::Rank>(peer),
              mpi_tag,
              world,
              req);
        },
        [](MPI_Status /*unused*/, fmpi::Ticket) {
          std::cout << "callback fire for send\n";
        });

    FMPI_DBG(rticket.id);
    FMPI_DBG(sticket.id);
  }

  auto consumer = std::async(
      std::launch::async,
      [&ready_tasks, &rbuf, &buf_alloc, ntasks = world.size()]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto n = ntasks;
        while ((n--) != 0u) {
          FMPI_DBG(buf_alloc.allocatedBlocks());
          FMPI_DBG(buf_alloc.allocatedHeapBlocks());
          FMPI_DBG(buf_alloc.isFull());
          auto ready = ready_tasks.value_pop();

          throw std::runtime_error("test");

          auto const [peer, s] = ready;

          FMPI_DBG(peer);
          FMPI_DBG_RANGE(s.begin(), s.end());
          // copy data
          std::copy(
              s.begin(), s.end(), std::next(rbuf.begin(), peer * blocksize));
          // release memory
          buf_alloc.dispose(s.data());
        }
      });

  dispatcher.loop_until_done();

  try {
    consumer.get();
  } catch (...) {
    std::cout << "exception catch block\n";
  }

  std::cout << "consumer done...\n";

  FMPI_DBG_RANGE(std::begin(rbuf), std::end(rbuf));

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  std::cout << "success...\n";

  return 0;
}
