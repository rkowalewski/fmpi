#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Config.hpp>
#include <fmpi/Dispatcher.hpp>
#include <fmpi/Span.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/container/StackContainer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/detail/Async.hpp>
#include <rtlx/Trace.hpp>
#include <tlx/container/ring_buffer.hpp>
#include <utility>

namespace fmpi {
namespace detail {

template <class Queue>
class NProducer {
 public:
  using value_type = typename Queue::value_type;

  NProducer(std::shared_ptr<Queue> channel, std::size_t n)
    : channel_(std::move(channel))
    , count_(n) {
  }

  bool operator()(value_type const& val) {
    if (count_ == 0u) {
      return false;
    }
    channel_->push(val);
    --count_;
    return true;
  }

 private:
  std::shared_ptr<Queue> channel_;
  std::size_t            count_;
};

template <class Queue>
class NConsumer {
 public:
  using value_type = typename Queue::value_type;

  NConsumer(std::shared_ptr<Queue> channel, std::size_t n)
    : channel_(std::move(channel))
    , count_(n) {
  }

  bool operator()(value_type& val) {
    if (count_ == 0u) {
      return false;
    }
    val = channel_->value_pop();
    --count_;
    return true;
  }

 private:
  std::shared_ptr<Queue> channel_;
  std::size_t            count_;
};

#if 0
template <class Channel>
class Scheduler {
  using task = typename Channel::value_type;

 public:
  Scheduler(std::shared_ptr<Channel> channel)
    : channel_(channel) {
  }

  template <class F, class OutputIterator>
  Iterator apply(F&& f, OutputIterator dest) {
  }

 private:
  std::shared_ptr<Channel> channel_;
};
#endif

}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void RingWaitsomeOverlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  auto const& config = Config::instance();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();
  // auto const me = ctx.rank();

  std::ostringstream os;
  os << Schedule::NAME << "WaitsomeOverlap" << NReqs;

  auto trace = rtlx::Trace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::Ring_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  rtlx::TimeTrace t_comm{trace, COMMUNICATION};

  Schedule const commAlgo{};

  // Number of Pipeline Stages
  constexpr std::size_t n_pipelines        = 2;
  constexpr std::size_t messages_per_round = 2;

  auto const n_exchanges  = ctx.size() - 1;
  auto const n_rounds     = commAlgo.phaseCount(ctx);
  auto const reqsInFlight = std::min(std::size_t(n_rounds), NReqs);
  auto const winsz        = reqsInFlight * messages_per_round;

  // intermediate buffer for two pipelines
  using buffer_allocator = HeapAllocator<value_type, true /*thread_safe*/>;

  std::size_t const nthreads = config.num_threads;

  FMPI_DBG(nthreads);
  FMPI_DBG(blocksize);

  auto const required = winsz * blocksize * n_pipelines;
  auto const capacity =
      nthreads * kMaxContiguousBufferSize / sizeof(value_type);

  auto const n_buffer_nels = std::min<std::size_t>(
      std::min(required, capacity),
      std::numeric_limits<typename buffer_allocator::index_type>::max());

  //FMPI_ASSERT(required <= capacity);

  auto buf_alloc = buffer_allocator{
      static_cast<typename buffer_allocator::index_type>(n_buffer_nels)};

  FMPI_DBG(n_buffer_nels);

  // queue for ready tasks
  using chunk = std::pair<mpi::Rank, Span<value_type>>;
#if 0
  auto received_chunks = boost::lockfree::spsc_queue<chunk>{n_rounds};
  using channel_t = boost::lockfree::spsc_queue<chunk>;
#else
  using channel_t = buffered_channel<chunk>;
#endif

  auto received_chunks = std::make_shared<channel_t>(n_rounds);

  fmpi::CommDispatcher<mpi::testsome> dispatcher{winsz};

  dispatcher.register_signal(
      fmpi::request_type::IRECV,
      [&buf_alloc, blocksize](
          fmpi::Message& message, MPI_Request & /*req*/) -> int {
        // allocator some buffer
        auto* buffer = buf_alloc.allocate(blocksize);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, blocksize);

        return 0;
      });

  dispatcher.register_signal(
      fmpi::request_type::IRECV,
      [](fmpi::Message& message, MPI_Request& req) -> int {
        auto ret = mpi::irecv(
            message.writable_buffer(),
            message.count(),
            message.type(),
            message.peer(),
            message.tag(),
            message.comm(),
            &req);

        FMPI_ASSERT(ret == MPI_SUCCESS);

        return ret;
      });

  dispatcher.register_signal(
      fmpi::request_type::ISEND,
      [](fmpi::Message& message, MPI_Request& req) -> int {
        auto ret = mpi::isend(
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
      fmpi::request_type::IRECV,
      [produce = detail::NProducer{received_chunks, n_exchanges}](
          fmpi::Message& message, MPI_Status const& status) mutable {
        FMPI_ASSERT(status.MPI_ERROR == MPI_SUCCESS);

        auto span = fmpi::make_span(
            static_cast<value_type*>(message.writable_buffer()),
            message.count());

        produce(std::make_pair(message.peer(), span));
      });

  dispatcher.start_worker();
  dispatcher.pinToCore(config.dispatcher_core);

  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;

  struct ComputeTime {
    duration queue{0};
    duration comp{0};
    duration total{0};
    duration schedule{0};
  } t_compute;

  FMPI_DBG("Sending Messages...");

  {
    timer t{t_compute.schedule};

    for (auto&& r : fmpi::range(commAlgo.phaseCount(ctx))) {
      auto const rpeer = commAlgo.recvRank(ctx, r);
      auto const speer = commAlgo.sendRank(ctx, r);

      if (rpeer != ctx.rank()) {
        auto recv_message = fmpi::Message{rpeer, EXCH_TAG_RING, ctx};

        dispatcher.dispatch(
            fmpi::request_type::IRECV, std::move(recv_message));
      }

      if (speer != ctx.rank()) {
        auto send_message = fmpi::Message(
            fmpi::make_span(std::next(begin, speer * blocksize), blocksize),
            speer,
            EXCH_TAG_RING,
            ctx);

        dispatcher.dispatch(
            fmpi::request_type::ISEND, std::move(send_message));
      }
    }
  }

  FMPI_DBG("dispatch done...");

  {
    timer t{t_compute.total};
    // chunks to merge
    std::vector<std::pair<OutputIt, OutputIt>> chunks;
    std::vector<std::pair<OutputIt, OutputIt>> processed;

    auto consume = detail::NConsumer{received_chunks, n_exchanges};

    chunks.reserve(ctx.size());
    // local task
    chunks.emplace_back(
        std::next(begin, ctx.rank() * blocksize),
        std::next(begin, (ctx.rank() + 1) * blocksize));

    processed.reserve(ctx.size());

    auto enough_work = [&chunks]() -> bool {
      // minimum number of chunks to merge: ideally we have a full level2
      // cache
      constexpr auto op_threshold = (NReqs / 2);
      return chunks.size() >= op_threshold;
    };

    // prefix sum over all processed chunks
    auto d_first = out;

    {
      timer t{t_compute.queue};
      chunk task{};
      while (consume(task)) {
        auto span = task.second;
        chunks.emplace_back(span.begin(), span.end());

        if (enough_work()) {
          timer t{t_compute.comp};
          // merge all chunks
          auto d_last = op(chunks, d_first);

          auto alloc_it = std::next(std::begin(chunks), processed.empty());

          for (; alloc_it != std::end(chunks); ++alloc_it) {
            FMPI_DBG_STREAM("release p " << alloc_it->first);
            buf_alloc.deallocate(alloc_it->first);
          }

          chunks.clear();

          // 3) increase out iterator
          processed.emplace_back(d_first, d_last);
          std::swap(d_first, d_last);
        } else {
          // TODO(rkowalewski): merge processed chunks
        }
      }
    }

    {
      timer      t{t_compute.comp};
      auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

      using merge_buffer_t = tlx::
          SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

      auto mergeBuffer = merge_buffer_t{nels};

      FMPI_DBG(chunks.size());

      // generate pairs of chunks to merge
      std::copy(
          std::begin(processed),
          std::end(processed),
          std::back_inserter(chunks));

      FMPI_DBG(chunks.size());

      auto const last = op(chunks, mergeBuffer.begin());

      for (auto f = std::next(std::begin(chunks), processed.empty());
           f != std::prev(std::end(chunks), processed.size());
           ++f) {
        FMPI_DBG_STREAM("release p " << f->first);
        buf_alloc.deallocate(f->first);
      }

      FMPI_ASSERT(last == mergeBuffer.end());

      FMPI_DBG(processed);

      std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
    }
  }

#ifndef NDEBUG
  dispatcher.loop_until_done();
  auto const stats = dispatcher.stats();

  trace.add_time("DispatcherThread.dispatch_time", stats.dispatch_time);
  trace.add_time("DispatcherThread.queue_time", stats.queue_time);
  trace.add_time("DispatcherThread.completion_time", stats.completion_time);
  trace.put(
      "DispatcherThread.iterations", static_cast<int>(stats.iterations));
#endif

  trace.add_time("ComputeThread.queue_time", t_compute.queue);
  trace.add_time("ComputeThread.compute_time", t_compute.comp);
  trace.add_time("ComputeThread.total_time", t_compute.total);
  trace.add_time("ScheduleThread", t_compute.schedule);
}
}  // namespace fmpi
#endif
