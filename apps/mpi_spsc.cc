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

  if (world.size() != 2) return -1;

//  using result_t = int;
//  using buffer_t = fmpi::Span<int>;
//  using rank_t   = mpi::Rank;
//  using tag_t    = int;

  int one = 1;
  int buf = 100;

  fmpi::CommDispatcher<int> dispatcher{world, 2};

  dispatcher.start_worker();

  // sleep for one second
  std::this_thread::sleep_for(std::chrono::seconds(1));

  FMPI_DBG(&buf);

  auto ready_tasks = fmpi::ThreadsafeQueue<fmpi::Span<int>>{};

  constexpr int mpi_tag = 0;

  mpi::Rank peer    = static_cast<mpi::Rank>(1) - world.rank();
  auto      rticket = dispatcher.postAsyncRecv(
      fmpi::make_span(&buf, 1),
      peer,
      mpi_tag,
      [me = world.rank(), &ready_tasks](
          fmpi::Ticket ticket, MPI_Status status, fmpi::Span<int> data) {
        std::ostringstream os;
        os << "[ Rank: " << me << "] [Ticket: " << ticket.id
           << " ] { status.MPI_SOURCE = " << status.MPI_SOURCE << ", "
           << "status.MPI_TAG = " << status.MPI_TAG << ", "
           << "status.MPI_ERROR = " << status.MPI_ERROR << "}\n";

        std::cout << os.str();
        ready_tasks.push_front(data);
      });

  auto sticket = dispatcher.postAsyncSend(
      fmpi::make_span(&one, 1),
      peer,
      mpi_tag,
      [](fmpi::Ticket, MPI_Status, fmpi::Span<int>) {
        std::cout << "callback fire for send\n";
      });

  std::cout << "enqueued ticket" << rticket.id << "\n";
  std::cout << "enqueued ticket" << sticket.id << "\n";

  auto consumer =
      std::async(std::launch::async, [&ready_tasks, ntasks = 1]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto n = ntasks;
        while (n--) {
          fmpi::Span<int> data;
          ready_tasks.pop_back(data);
          std::ostringstream os;
          os << "received chunk: " << data.size() << " nels\n";
          std::cout << os.str();
        }
      });

  dispatcher.loop_until_done();
  std::cout << "dispatcher done...waiting for consumer\n";
  consumer.wait();

  std::cout << "consumer done...\n";

  FMPI_DBG(buf);

  if (buf == 1 && world.rank() == 0) {
    std::cout << "test successful\n";
  }

  return 0;
}
