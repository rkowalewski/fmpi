#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Config.hpp>
#include <fmpi/Dispatcher.hpp>
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

template <class Channel, class T = typename Channel::value_type>
class Producer {
 public:
  using channel    = Channel;
  using value_type = T;

  Producer() = default;

  Producer(std::shared_ptr<channel> chan) noexcept
    : channel_(std::move(chan)) {
  }

  void push(value_type const& val) {
    channel_->push(val);
  }

 private:
  std::shared_ptr<channel> channel_;
};

template <class Channel, class T = typename Channel::value_type>
class Consumer {
 public:
  using channel    = Channel;
  using value_type = T;

  Consumer() = default;

  Consumer(std::shared_ptr<channel> chan) noexcept
    : channel_(std::move(chan)) {
  }

  void pop(value_type& val) {
    FMPI_ASSERT(channel_->front());
    val = std::move(*(channel_->front()));
    channel_->pop();
  }

 private:
  std::shared_ptr<channel> channel_;
};

template <class T>
class Consumer<buffered_channel<T>> {
 public:
  using value_type = T;
  using channel    = buffered_channel<value_type>;

  Consumer() = default;

  Consumer(std::shared_ptr<channel> chan) noexcept
    : channel_(std::move(chan)) {
  }

  void pop(value_type& val) {
    val = channel_->value_pop();
  }

 private:
  std::shared_ptr<channel> channel_;
};

template <class Channel>
class NProducer {
  using producer = Producer<Channel>;

 public:
  using value_type = typename Channel::value_type;

  NProducer(std::shared_ptr<Channel> channel, std::size_t n) noexcept
    : channel_(std::move(channel))
    , count_(n) {
  }

  bool operator()(value_type const& val) {
    if (count_ == 0u) {
      return false;
    }
    channel_.push(val);
    --count_;
    return true;
  }

 private:
  producer    channel_;
  std::size_t count_;
};

template <class Channel>
class NConsumer {
  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;
  using consumer = Consumer<Channel>;

 public:
  using value_type = typename Channel::value_type;

  NConsumer(std::shared_ptr<Channel> channel, std::size_t n) noexcept
    : channel_(std::move(channel))
    , count_(n) {
  }

  bool operator()(value_type& val) {
    timer{time_};
    if (count_ == 0u) {
      return false;
    }
    channel_.pop(val);
    --count_;
    return true;
  }

  duration time() const noexcept {
    return time_;
  }

 private:
  consumer    channel_{};
  std::size_t count_{};
  duration    time_{0};
};

}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void ring_waitsome_overlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;

  struct Times {
    duration comm_enqueue{0};
    // comm_dequeue in dispatcher...

    // comp_enqueue in dispatcher...
    duration comp_dequeue{0};

    duration comp_compute{0};
    duration idle{0};
  };

  struct Times times {};

  auto const& config = Config::instance();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();

  std::ostringstream os;
  os << Schedule::NAME << "WaitsomeOverlap" << NReqs;

  auto trace = rtlx::Trace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  Schedule const commAlgo{};

  // Each round is composed of an isend-irecv pair...
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

  // Number of Pipeline Stages
  constexpr std::size_t n_pipelines = 2;

  auto const required = reqsInFlight * blocksize * n_pipelines;
  auto const capacity =
      nthreads * kMaxContiguousBufferSize / sizeof(value_type);

  auto const n_buffer_nels = std::min<std::size_t>(
      std::min(required, capacity),
      std::numeric_limits<typename buffer_allocator::index_type>::max());

  auto buf_alloc = buffer_allocator{
      static_cast<typename buffer_allocator::index_type>(n_buffer_nels)};

  FMPI_DBG(n_buffer_nels);

  // queue for ready tasks
  using chunk = std::pair<mpi::Rank, gsl::span<value_type>>;
#if 0
  auto received_chunks = boost::lockfree::spsc_queue<chunk>{n_rounds};
  using channel_t = boost::lockfree::spsc_queue<chunk>;
#else
  using channel_t = buffered_channel<chunk>;
#endif

  auto received_chunks = std::make_shared<channel_t>(n_rounds);

  auto comm_channel =
      std::make_shared<CommChannel>(n_exchanges * messages_per_round);

  auto dispatcher = CommDispatcher<mpi::testsome>{comm_channel, winsz};

  dispatcher.register_signal(
      message_type::IRECV,
      [&buf_alloc, blocksize](
          Message& message, MPI_Request & /*req*/) -> int {
        // allocator some buffer
        auto* buffer = buf_alloc.allocate(blocksize);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, blocksize);

        return 0;
      });

  dispatcher.register_signal(
      message_type::IRECV, [](Message& message, MPI_Request& req) -> int {
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
      message_type::ISEND, [](Message& message, MPI_Request& req) -> int {
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
      message_type::IRECV,
      [callback = detail::NProducer{
           received_chunks,
           n_exchanges,
       }](Message& message /*, MPI_Status const& status*/) mutable {
#if 0
        FMPI_ASSERT(status.MPI_ERROR == MPI_SUCCESS);
        FMPI_ASSERT(status.MPI_SOURCE == message.peer());
        FMPI_ASSERT(status.MPI_TAG == message.tag());
#endif
        auto span = gsl::span(
            static_cast<value_type*>(message.writable_buffer()),
            message.count());

        auto ret = callback(std::make_pair(message.peer(), span));
        FMPI_ASSERT(ret);
      });

  dispatcher.start_worker();
  dispatcher.pinToCore(config.dispatcher_core);

  {
    FMPI_DBG("Sending Messages");

    timer{times.comm_enqueue};

    for (auto&& r : range(commAlgo.phaseCount(ctx))) {
      auto const rpeer = commAlgo.recvRank(ctx, r);
      auto const speer = commAlgo.sendRank(ctx, r);

      if (rpeer != ctx.rank()) {
        auto recv = Message{rpeer, EXCH_TAG_RING, ctx};

        comm_channel->enqueue(CommTask{message_type::IRECV, recv});
      }

      if (speer != ctx.rank()) {
        auto span = gsl::span(std::next(begin, speer * blocksize), blocksize);

        auto send = Message{span, speer, EXCH_TAG_RING, ctx};

        comm_channel->enqueue(CommTask{message_type::ISEND, send});
      }
    }
  }

  {
    FMPI_DBG("processing message arrivals...");

    // chunks to merge
    std::vector<std::pair<OutputIt, OutputIt>> chunks;
    std::vector<std::pair<OutputIt, OutputIt>> processed;

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
      auto consumer = detail::NConsumer{received_chunks, n_exchanges};

      chunk task{};
      while (consumer(task)) {
        auto span = task.second;
        chunks.emplace_back(span.data(), span.data() + span.size());

        if (enough_work()) {
          timer{times.comp_compute};
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
      times.comp_dequeue = consumer.time();
    }

    {
      timer{times.comp_compute};
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

  {
    timer{times.idle};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    dispatcher.loop_until_done();
  }

  auto const stats = dispatcher.stats();

  trace.add_time("Tcomm.enqueue", times.comm_enqueue);
  trace.add_time("Tcomm.dequeue", stats.queue_time);

  trace.add_time("Tcomp.enqueue", stats.callback_time);
  trace.add_time("Tcomp.dequeue", times.comp_dequeue);

  trace.add_time(COMPUTATION, times.comp_compute);
  trace.add_time("Tcomp.idle", times.idle);

  trace.add_time("Tcomm.dispatch", stats.dispatch_time);
  trace.add_time("Tcomm.progress", stats.completion_time);

  // other
  trace.put("Tcomm.iterations", static_cast<int>(stats.iterations));
}
}  // namespace fmpi
#endif
