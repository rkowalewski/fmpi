#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <future>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include <fmpi/NumericRange.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/ScopedLambda.hpp>

#include <boost/lockfree/spsc_queue.hpp>

constexpr std::size_t N         = 100;
constexpr std::size_t threshold = 10;

using Message = int;
using MessageQueue =
    boost::lockfree::spsc_queue<Message, boost::lockfree::capacity<100>>;

// The producer functionality
std::vector<Message> Producer(
    MessageQueue& queue, const size_t enough_processed)
{
  std::vector<Message> all_data;
  all_data.resize(enough_processed);

  std::iota(all_data.begin(), all_data.end(), 0);

  std::vector<Message> pushed_data;  // for later verification
  for (Message m : all_data) {
    while (!queue.push(m)) {
      // std::this_thread::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  return pushed_data;
}

int main(int argc, char* argv[])
{
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "MPI_THREAD_FUNNELED not supported\n";
    return 1;
  }

  int f;
  MPI_Is_thread_main(&f);

  if (f == 0) {
    std::cerr << "is not main thread\n";
    return 1;
  }

  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  auto q = MessageQueue{};

  auto consumer = std::async(std::launch::async, [&q]() {
    std::vector<Message> recv;

    std::size_t nrecv = 0;

    while (nrecv < N) {
      while (q.read_available() < threshold) {
        std::this_thread::yield();
      }

      auto nr = q.consume_all([&recv](auto v) { recv.emplace_back(v); });

      std::ostringstream os;
      os << "received " << nr << " messages:\n";

      std::copy(
          std::next(std::begin(recv), nrecv),
          std::next(std::begin(recv), nrecv + nr),
          std::ostream_iterator<Message>(os, ", "));

      os << "\n";

      std::cout << os.str();

      nrecv += nr;
    }

    return recv.size();
  });

  Producer(q, N);

  auto ret = consumer.get();

  std::cout << "consumer processed " << ret << "items\n";
  return 0;
}
