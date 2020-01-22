#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <cstdlib>

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

int run();

constexpr std::size_t NCPU = 48;
constexpr std::size_t NCOMM_CORES = 1;

struct CorePinning {
  int mpi_core;
  int dispatcher_core;
  int scheduler_core;
  int comp_core;
};

CorePinning config_pinning(std::size_t domain_size);

void (std::promise<void> promise, mpi::Context const& ctx, std::size_t message_size) {

}

int main(int argc, char* argv[]) {
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  if (provided < MPI_THREAD_SERIALIZED) {
    std::cerr << "MPI_THREAD_SERIALIZED not supported\n";
    return 1;
  }

  run();

  return 0;
}

int run() {

  auto const domain_size = static_cast<std::size_t>(std::atoll(std::getenv("FMPI_DOMAIN_SIZE")));

  mpi::Context const world{MPI_COMM_WORLD};

  auto const pinning = config_pinning(domain_size);
  FMPI_DBG(pinning);

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
  //dispatcher.pinToCore(dispatcher_core);

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

  std::this_thread::sleep_for(std::chrono::seconds(2));

  auto n = world.size();

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

#if 0
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

          auto const [peer, s] = ready;

          // copy data
          std::copy(
              s.begin(), s.end(), std::next(rbuf.begin(), peer * blocksize));
          // release memory
          buf_alloc.dispose(s.data());
        }
      });
#endif

  dispatcher.loop_until_done();

#if 0
  try {
    consumer.get();
  } catch (...) {
    std::cout << "exception catch block\n";
  }
  std::cout << "consumer done...\n";
#endif


  FMPI_DBG_RANGE(std::begin(rbuf), std::end(rbuf));

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  std::cout << "success...\n";

  return 0;
}

CorePinning config_pinning(std::size_t domain_size) {
  {
    int flag;
    FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));
    FMPI_ASSERT(flag);
  }

  CorePinning pinning;

  auto const my_core = sched_getcpu();
  auto const n_comp_cores = domain_size - NCOMM_CORES;
  auto const domain_id = (my_core % NCPU) / domain_size;
  auto const is_rank_on_comm = (my_core % domain_size) == 0;


  if (is_rank_on_comm) {
    //MPI rank is on the communication core. So we dispatch on the other
    //hyperthread
    pinning.dispatcher_core = (my_core < NCPU) ? my_core + NCPU : my_core - NCPU;
    pinning.scheduler_core = my_core;
    pinning.comp_core = pinning.scheduler_core + 1;
  } else {
    //MPI rank is somewhere in the Computation Domain, so we just take the
    //communication core.
    pinning.dispatcher_core = domain_id * domain_size;
    pinning.scheduler_core = pinning.dispatcher_core + NCPU;
    pinning.comp_core = my_core;
  }

  pinning.mpi_core = my_core;

  return pinning;
}

std::ostream& operator<<(std::ostream& os, const CorePinning& pinning) {
  os << "{ rank: " << pinning.mpi_core;
  os << ", scheduler: " << pinning.scheduler_core;
  os << ", dispatcher: " << pinning.dispatcher_core;
  os << ", comp: " << pinning.comp_core;
  os << " }";
  return os;
}
