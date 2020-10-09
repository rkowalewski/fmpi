#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <boost/lockfree/spsc_queue.hpp>
#include <cassert>
#include <cstdlib>
#include <fmpi/Debug.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/concurrency/Async.hpp>
#include <fmpi/concurrency/BufferedChannel.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/Request.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <future>
#include <gsl/span>
#include <iostream>
#include <numeric>
#include <random>
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
  int provided = 0;
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

  auto const& pinning = fmpi::Pinning::instance();

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

  auto buf_alloc = fmpi::ThreadAllocator<value_type>{};

  auto f_comp = fmpi::async(
      0 /*unused*/,
      [&ready_tasks, &rbuf, &buf_alloc, ntasks = world.size()]() -> iterator {
        auto n = ntasks;
        while ((n--) != 0u) {
          auto ready = ready_tasks.value_pop();

          auto const [peer, s] = ready;

          // copy data
          std::copy(
              s.begin(), s.end(), std::next(rbuf.begin(), peer * blocksize));
          // release memory
          buf_alloc.deallocate(s.data(), blocksize);
        }
        return rbuf.end();
      });

  // make context
  std::array<std::size_t, fmpi::detail::n_types> nslots{};
  nslots.fill(winsz / 2);

  fmpi::collective_promise promise;
  auto                     future = promise.get_future();

  auto schedule_ctx =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

  fmpi::CommDispatcher dispatcher{};

  schedule_ctx->register_signal(
      fmpi::message_type::IRECV, [&buf_alloc](fmpi::Message& message) {
        // allocator some buffer
        auto* b = buf_alloc.allocate(blocksize);
        message.set_buffer(gsl::span(b, blocksize));
      });

  schedule_ctx->register_callback(
      fmpi::message_type::IRECV,
      [&ready_tasks](std::vector<fmpi::Message> msgs) {
        for (auto&& msg : msgs) {
          auto span =
              gsl::span(static_cast<value_type*>(msg.buffer()), msg.count());

          ready_tasks.push(std::make_pair(msg.peer(), span));
        }
      });

  // submit into dispatcher
  auto hdl = dispatcher.submit(std::move(schedule_ctx));

  for (auto&& peer : fmpi::range(mpi::Rank(0), mpi::Rank(world.size()))) {
    constexpr int mpi_tag = 123;
    auto recv_message     = fmpi::Message{peer, mpi_tag, world.mpiComm()};

    dispatcher.schedule(hdl, fmpi::message_type::IRECV, recv_message);

    auto send_message = fmpi::Message(
        gsl::span(&*std::next(std::begin(sbuf), peer * blocksize), blocksize),
        peer,
        mpi_tag,
        world.mpiComm());

    dispatcher.schedule(hdl, fmpi::message_type::ISEND, send_message);
  }

  iterator ret;
  try {
    ret = f_comp.get();
  } catch (...) {
    std::cout << "computation done...\n";
  }

  assert(ret == rbuf.end());

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  return 0;
}
