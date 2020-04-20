#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <cstdlib>

#include <gsl/span>

#include <boost/container/small_vector.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <fmpi/Dispatcher.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/allocator/ContiguousPoolAllocator.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/container/BoundedBuffer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/detail/Async.hpp>
#include <fmpi/detail/Capture.hpp>
#include <fmpi/mpi/Algorithm.hpp>
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

int run();

int main(int argc, char* argv[]) {
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::Context const world{MPI_COMM_WORLD};

  int nplaces = omp_get_num_places();
  printf("omp_get_num_places: %d\n", nplaces);

  std::vector<int> myprocs;
  for (int i = 0; i < nplaces; ++i) {
    int nprocs = omp_get_place_num_procs(i);
    myprocs.resize(nprocs);
    omp_get_place_proc_ids(i, myprocs.data());
    std::ostringstream os;
    os << "rank: " << world.rank() << ", place num: " << i << ", places: {";
    std::copy(
        myprocs.begin(), myprocs.end(), std::ostream_iterator<int>(os, ", "));
    os << "}\n";
    std::cout << os.str();
  }

  if (provided < MPI_THREAD_SERIALIZED) {
    std::cerr << "MPI_THREAD_SERIALIZED not supported\n";
    return 2;
  }

  run();

  return 0;
}

int run() {
  mpi::Context const world{MPI_COMM_WORLD};

  auto const& pinning = fmpi::Config::instance();

  FMPI_DBG(pinning);

  constexpr std::size_t winsz     = 4;
  constexpr std::size_t blocksize = 1;

  using value_type = int;
  using container  = std::vector<value_type>;
  using iterator   = typename container::iterator;

  container sbuf(world.size() * blocksize);
  container rbuf(world.size() * blocksize);

  int const first = world.rank() * world.size();
  std::iota(std::begin(sbuf), std::end(sbuf), first);

  container expect(world.size() * blocksize);

  int val = world.rank();
  std::generate(
      std::begin(expect),
      std::end(expect),
      [&val, size = world.size()]() mutable {
        return std::exchange(val, val + size);
      });

  FMPI_DBG_RANGE(std::begin(sbuf), std::end(sbuf));

  // Pipeline 2
  using rank_data_pair = std::pair<mpi::Rank, gsl::span<value_type>>;

  auto ready_tasks = fmpi::buffered_channel<rank_data_pair>{world.size()};

  uint16_t const size = std::min<uint16_t>(winsz, world.size()) * blocksize;

  auto buf_alloc =
      fmpi::HeapAllocator<value_type, true /*thread_safe*/>{size};

  auto const nreqs = 2 * world.size();

  using comm_channel = typename fmpi::CommChannel;
  auto channel       = std::make_shared<comm_channel>(nreqs);

  // Dispatcher Thread
  fmpi::CommDispatcher dispatcher{channel, winsz};

  dispatcher.register_signal(
      fmpi::message_type::IRECV,
      [&buf_alloc](fmpi::Message& message, MPI_Request& /*req*/) {
        // allocator some buffer
        auto* b = buf_alloc.allocate(blocksize);
        message.set_buffer(b, blocksize);

        return 0;
      });

  dispatcher.register_signal(
      fmpi::message_type::IRECV,
      [](fmpi::Message& message, MPI_Request& req) -> int {
        return MPI_Irecv(
            message.writable_buffer(),
            message.count(),
            message.type(),
            message.peer(),
            message.tag(),
            message.comm(),
            &req);
      });

  dispatcher.register_signal(
      fmpi::message_type::ISEND,
      [](fmpi::Message& message, MPI_Request& req) -> int {
        auto ret = MPI_Isend(
            message.readable_buffer(),
            message.count(),
            message.type(),
            message.peer(),
            message.tag(),
            message.comm(),
            &req);

        FMPI_ASSERT(ret == MPI_SUCCESS);
        return ret;
      });

  dispatcher.register_callback(
      fmpi::message_type::IRECV,
      [&ready_tasks](fmpi::Message& message /*, MPI_Status const& status*/) {
        auto span = gsl::span(
            static_cast<value_type*>(message.writable_buffer()),
            message.count());

        ready_tasks.push(std::make_pair(message.peer(), span));
      });

  dispatcher.start_worker();
  dispatcher.pinToCore(pinning.dispatcher_core);

  auto f_comm = fmpi::async(
      pinning.scheduler_core,
      [first   = std::begin(sbuf),
       last    = std::end(sbuf),
       channel = std::move(channel),
       &world]() {
        constexpr int mpi_tag = 123;

        for (auto&& peer :
             fmpi::range(mpi::Rank(0), mpi::Rank(world.size()))) {
          auto recv_message = fmpi::Message{peer, mpi_tag, world};

          channel->enqueue(fmpi::CommTask{fmpi::message_type::IRECV,
                                          recv_message});

          auto send_message = fmpi::Message(
              gsl::span(&first[peer], blocksize), peer, mpi_tag, world);

          channel->enqueue(fmpi::CommTask{fmpi::message_type::ISEND,
                                          send_message});
        }
      });

  auto f_comp = fmpi::async(
      pinning.comp_core,
      [&ready_tasks, &rbuf, &buf_alloc, ntasks = world.size()]() -> iterator {
        auto n = ntasks;
        while ((n--) != 0u) {
          FMPI_DBG(buf_alloc.allocatedBlocks());
          FMPI_DBG(buf_alloc.allocatedHeapBlocks());
          FMPI_DBG(buf_alloc.isFull());
          auto ready = ready_tasks.value_pop();

          auto const [peer, s] = ready;

          // copy data
          std::copy(
              s.begin(), s.end(), std::next(rbuf.begin(), peer * blocksize));
          // release memory
          buf_alloc.dispose(s.data());
        }
        return rbuf.end();
      });

  iterator ret;
  try {
    ret = f_comp.get();
    f_comm.wait();
  } catch (...) {
    std::cout << "computation done...\n";
  }

  FMPI_ASSERT(ret == rbuf.end());

  // auto const pending_tasks = dispatcher.pendingTasks();
  // FMPI_ASSERT(pending_tasks.first == 0 && pending_tasks.second == 0);
  // FMPI_ASSERT(pending_tasks.first == 0 && pending_tasks.second == 0);

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  dispatcher.loop_until_done();

  return 0;
}
