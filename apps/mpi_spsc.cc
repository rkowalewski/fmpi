#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <boost/lockfree/spsc_queue.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
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
  fmpi::CommDispatcher<int> dispatcher{world, winsz};

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

  // sleep for one second
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // FMPI_DBG(&buf);

  auto ready_tasks = fmpi::ThreadsafeQueue<fmpi::Span<int>>{};

  constexpr int mpi_tag = 0;

  for (auto&& peer : fmpi::range(world.size())) {
    auto rticket = dispatcher.postAsyncRecv(
        fmpi::make_span(&rbuf[peer], blocksize),
        static_cast<mpi::Rank>(peer),
        mpi_tag,
        [me = world.rank(), &ready_tasks](
            fmpi::Ticket, MPI_Status status, fmpi::Span<int> data) {
          FMPI_CHECK(status.MPI_ERROR == MPI_SUCCESS);
          ready_tasks.push_front(data);
        });

    auto sticket = dispatcher.postAsyncSend(
        fmpi::make_span(&sbuf[peer], blocksize),
        static_cast<mpi::Rank>(peer),
        mpi_tag,
        [](fmpi::Ticket, MPI_Status, fmpi::Span<int>) {
          std::cout << "callback fire for send\n";
        });

    FMPI_DBG(rticket.id);
    FMPI_DBG(sticket.id);
  }

  auto consumer =
      std::async(std::launch::async, [&ready_tasks, ntasks = world.size()]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto n = ntasks;
        while (n--) {
          fmpi::Span<int> data;
          ready_tasks.pop_back(data);
        }
      });

  dispatcher.loop_until_done();
  std::cout << "dispatcher done...waiting for consumer\n";
  consumer.wait();

  std::cout << "consumer done...\n";

  FMPI_DBG_RANGE(std::begin(rbuf), std::end(rbuf));

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  std::cout << "success...\n";

  return 0;
}
