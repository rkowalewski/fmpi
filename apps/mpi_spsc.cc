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

struct CorePinning {
  int mpi_core        = -1;
  int dispatcher_core = -1;
  int scheduler_core  = -1;
  int comp_core       = -1;
};

CorePinning config_pinning();

int run();

template <typename RET, typename FUNC, typename... ARGS>
constexpr void call_with_promise(
    fmpi::Capture<RET, FUNC, ARGS...>& callable, std::promise<RET>& pr) {
  pr.set_value(callable());
}

template <typename FUNC, typename... ARGS>
constexpr void call_with_promise(
    fmpi::Capture<void, FUNC, ARGS...>& callable, std::promise<void>& pr) {
  callable();
  pr.set_value();
}

template <typename R, typename F, typename... Ts>
inline std::future<R> async_on_core(
    int my_core, int their_core, F&& f, Ts&&... params) {
  using promise = std::promise<R>;

  auto lambda =
      fmpi::makeCapture<R>(std::forward<F>(f), std::forward<Ts>(params)...);

  auto pr = std::make_shared<promise>();

  if (my_core != their_core) {
    auto t = std::thread([fn = std::move(lambda), pr]() mutable {
      call_with_promise(fn, *pr);
    });

    fmpi::pinThreadToCore(t, their_core);
    t.detach();
  } else {
    call_with_promise(lambda, *pr);
  }

  return pr->get_future();
}

template <
    class ContiguousIter,
    class Dispatcher,
    class BufAlloc,
    class ReadyCallback>
void schedule_comm(
    // source buffer
    ContiguousIter first,
    ContiguousIter last,
    // MPI Context
    mpi::Context const& ctx,
    // length of each message
    std::size_t blocksize,
    Dispatcher& dispatcher,
    // Allocator for itermediate buffer
    BufAlloc&&      bufAlloc,
    ReadyCallback&& ready) {
  using value_type =
      typename std::iterator_traits<ContiguousIter>::value_type;

  constexpr int mpi_tag = 0;

  FMPI_CHECK(std::next(first, ctx.size() * blocksize) == last);

  FMPI_DBG(ctx.size());

  for (auto&& peer :
       fmpi::range(static_cast<mpi::Rank>(0), mpi::Rank(ctx.size()))) {
    auto rticket = dispatcher.postAsync(
        fmpi::request_type::IRECV,
        [cb = std::forward<BufAlloc>(bufAlloc), peer, &ctx](
            MPI_Request* req, fmpi::Ticket ticket) -> int {
          auto s = cb(ticket);
          FMPI_DBG_STREAM("mpi::irecv from rank " << peer);

          return mpi::irecv(s.data(), s.size(), peer, mpi_tag, ctx, req);
        },
        [cb = std::forward<ReadyCallback>(ready), peer](
            MPI_Status status, fmpi::Ticket ticket) {
          FMPI_CHECK(status.MPI_ERROR == MPI_SUCCESS);
          cb(ticket, peer);
        });

    auto sb = fmpi::Span<const value_type>(&first[peer], blocksize);

    auto sticket = dispatcher.postAsync(
        fmpi::request_type::ISEND,
        [sb, peer, &ctx](MPI_Request* req, fmpi::Ticket) -> int {
          return mpi::isend(
              sb.data(),
              sb.size(),
              static_cast<mpi::Rank>(peer),
              mpi_tag,
              ctx,
              req);
        },
        [](MPI_Status /*unused*/, fmpi::Ticket) {
          std::cout << "callback fire for send\n";
        });

    FMPI_DBG(rticket.id);
    FMPI_DBG(sticket.id);
  }
}

int main(int argc, char* argv[]) {
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  if (provided < MPI_THREAD_SERIALIZED) {
    std::cerr << "MPI_THREAD_SERIALIZED not supported\n";
    return 2;
  }

  run();

  return 0;
}

int run() {
  mpi::Context const world{MPI_COMM_WORLD};

  auto const pinning = config_pinning();

  FMPI_DBG(pinning);

  constexpr std::size_t winsz = 4;

  constexpr std::size_t blocksize = 1;

  using value_type = int;
  using container  = std::vector<value_type>;

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
  using rank_data_pair = std::pair<mpi::Rank, fmpi::Span<value_type>>;

  auto ready_tasks = fmpi::buffered_channel<rank_data_pair>{world.size()};

  uint16_t const size =
      std::min<uint16_t>(winsz, world.size()) * blocksize * 2;

  auto buf_alloc =
      fmpi::HeapAllocator<value_type, true /*thread_safe*/>{size};

  // Dispatcher Thread
  fmpi::CommDispatcher dispatcher{winsz};
  dispatcher.start_worker();
  dispatcher.pinToCore(pinning.dispatcher_core);

  using token_data_pair = std::pair<fmpi::Ticket, fmpi::Span<value_type>>;
  constexpr std::size_t n_pipelines = 2;
  boost::container::small_vector<token_data_pair, winsz * n_pipelines> blocks;

  auto enqueue = [&buf_alloc, &blocks](fmpi::Ticket ticket) {
    // allocator some buffer
    auto* b = buf_alloc.allocate(blocksize);
    auto  s = fmpi::make_span(b, blocksize);
    blocks.push_back(std::make_pair(ticket, s));
    return s;
  };

  auto dequeue = [&blocks, &ready_tasks](
                     fmpi::Ticket ticket, mpi::Rank peer) {
    auto it = std::find_if(
        std::begin(blocks), std::end(blocks), [ticket](const auto& v) {
          return v.first == ticket;
        });

    FMPI_CHECK(it != std::end(blocks));

    ready_tasks.push(std::make_pair(peer, it->second));
    blocks.erase(it);
  };

  auto f_comm = async_on_core<void>(
      pinning.mpi_core,
      pinning.scheduler_core,
      [first = std::begin(sbuf),
       last  = std::end(sbuf),
       world,
       &dispatcher,
       enq = std::move(enqueue),
       deq = std::move(dequeue)]() {
        schedule_comm(first, last, world, blocksize, dispatcher, enq, deq);
      });

  using return_type = typename container::iterator;
  auto f_comp       = async_on_core<return_type>(
      pinning.mpi_core,
      pinning.comp_core,
      [&ready_tasks, &rbuf, &buf_alloc, ntasks = world.size()]()
          -> return_type {
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

  return_type ret;
  try {
    ret = f_comp.get();
    f_comm.wait();
  } catch (...) {
    std::cout << "computation done...\n";
  }

  FMPI_ASSERT(ret == rbuf.end());

  auto const pending_tasks = dispatcher.pendingTasks();
  FMPI_ASSERT(pending_tasks.first == 0 && pending_tasks.second == 0);
  FMPI_ASSERT(pending_tasks.first == 0 && pending_tasks.second == 0);

  if (!std::equal(std::begin(rbuf), std::end(rbuf), std::begin(expect))) {
    throw std::runtime_error("invalid result");
  }

  std::cout << "success...\n";

  dispatcher.loop_until_done();

  return 0;
}

CorePinning config_pinning() {
  {
    int flag;
    FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));
    FMPI_ASSERT(flag);
  }

  std::size_t domain_size = 1;
  {
    auto const* env = std::getenv("FMPI_DOMAIN_SIZE");
    if (env) {
      std::istringstream{std::string(env)} >> domain_size;
    }
  }

  CorePinning pinning;

  auto const nthreads = std::thread::hardware_concurrency();
  FMPI_ASSERT(nthreads >= 4);

  auto const ncores = nthreads / 2;

  auto const my_core         = sched_getcpu();
  auto const domain_id       = (my_core % ncores) / domain_size;
  auto const is_rank_on_comm = (my_core % domain_size) == 0;

  if (domain_size == 1) {
    pinning.dispatcher_core = (my_core + 1) % nthreads;
    pinning.scheduler_core  = (my_core + 2) % nthreads;
    pinning.comp_core       = my_core;
  } else if (is_rank_on_comm) {
    // MPI rank is on the communication core. So we dispatch on the other
    // hyperthread
    pinning.dispatcher_core =
        (std::size_t(my_core) < ncores) ? my_core + ncores : my_core - ncores;
    pinning.scheduler_core = my_core;
    pinning.comp_core      = pinning.scheduler_core + 1;
  } else {
    // MPI rank is somewhere in the Computation Domain, so we just take the
    // communication core.
    pinning.dispatcher_core = domain_id * domain_size;
    pinning.scheduler_core  = pinning.dispatcher_core + ncores;
    pinning.comp_core       = my_core;
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
