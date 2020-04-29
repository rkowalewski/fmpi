#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Config.hpp>
#include <fmpi/Dispatcher.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/container/StackContainer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <rtlx/Trace.hpp>
#include <tlx/container/ring_buffer.hpp>
#include <utility>

namespace fmpi {

namespace detail {

template <class Timer>
class TimeTrace {
  using duration = typename Timer::duration;

  std::string name_;
  rtlx::Trace trace_;

 public:
  duration initialize{0};
  duration dispatch{0};
  duration receive{0};
  duration shutdown{0};
  duration compute{0};
  duration idle{0};

  TimeTrace(std::string name)
    : name_(std::move(name))
    , trace_(name_) {
  }

  ~TimeTrace() {
    trace_.add_time("Tmain.initialize", initialize);
    trace_.add_time("Tmain.dispatch", dispatch);
    trace_.add_time("Tmain.receive", receive);
    trace_.add_time("Tmain.shutdown", shutdown);
    trace_.add_time("Tmain.compute", compute);
    trace_.add_time("Tmain.idle", idle);
  }

  std::string_view name() const noexcept {
    return name_;
  }

  rtlx::Trace& trace() noexcept {
    return trace_;
  }
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
  using timer = rtlx::Timer<>;

  constexpr auto algorithm_name = std::string_view("WaitsomeOverlap");

  auto const& config = Config::instance();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();

  auto trace = detail::TimeTrace<timer>{std::string{Schedule::NAME} +
                                        std::string{algorithm_name} +
                                        std::to_string(NReqs)};

  FMPI_DBG_STREAM(
      "running algorithm " << trace.name() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace.trace());
    return;
  }

  timer t_init{trace.initialize};

  Schedule const commAlgo{};

  // Each round is composed of an isend-irecv pair...
  constexpr std::size_t messages_per_round = 2;

  auto const n_exchanges  = ctx.size() - 1;
  auto const n_messages   = n_exchanges * messages_per_round;
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
  //auto data_channel = boost::lockfree::spsc_queue<chunk>{n_rounds};
#else
  using channel_t = SPSCNChannel<chunk>;
#endif

  using dispatcher_t = CommDispatcher<mpi::waitsome>;

  auto comm_channel =
      std::make_shared<typename dispatcher_t::channel>(n_messages);

  auto data_channel = std::make_shared<channel_t>(n_exchanges);

  auto comm_dispatcher = dispatcher_t{comm_channel, winsz};

  comm_dispatcher.register_signal(
      message_type::IRECV,
      [&buf_alloc, blocksize](
          Message& message, MPI_Request & /*req*/) -> int {
        // allocator some buffer
        auto* buffer = buf_alloc.allocate(blocksize);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(gsl::span(buffer, blocksize));

        return 0;
      });

  comm_dispatcher.register_signal(
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

  comm_dispatcher.register_signal(
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

  comm_dispatcher.register_callback(
      message_type::IRECV,
      [data_channel](
          Message& message /*, MPI_Status const& status*/) mutable {
#if 0
        FMPI_ASSERT(status.MPI_ERROR == MPI_SUCCESS);
        FMPI_ASSERT(status.MPI_SOURCE == message.peer());
        FMPI_ASSERT(status.MPI_TAG == message.tag());
#endif
        auto span = gsl::span(
            static_cast<value_type*>(message.writable_buffer()),
            message.count());

        auto ret =
            data_channel->enqueue(std::make_pair(message.peer(), span));
        FMPI_ASSERT(ret);
      });

  comm_dispatcher.start_worker();
  comm_dispatcher.pinToCore(config.dispatcher_core);

  t_init.finish();

  {
    FMPI_DBG("Sending essages");
    timer{trace.dispatch};

    for (auto&& r : range(commAlgo.phaseCount(ctx))) {
      auto const rpeer = commAlgo.recvRank(ctx, r);
      auto const speer = commAlgo.sendRank(ctx, r);

      if (rpeer != ctx.rank()) {
        auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};

        comm_channel->enqueue(CommTask{message_type::IRECV, recv});
      }

      if (speer != ctx.rank()) {
        auto span = gsl::span(std::next(begin, speer * blocksize), blocksize);

        auto send = Message{span, speer, kTagRing, ctx.mpiComm()};

        comm_channel->enqueue(CommTask{message_type::ISEND, send});
      }
    }
  }

  {
    FMPI_DBG("processing message arrivals...");

    timer{trace.receive};

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
      chunk task{};
      while (data_channel->wait_dequeue(task)) {
        auto span = task.second;
        chunks.emplace_back(span.data(), span.data() + span.size());

        if (enough_work()) {
          timer{trace.compute};
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
      timer{trace.compute};
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
    timer{trace.idle};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    comm_dispatcher.loop_until_done();
  }

  timer t_shutdown{trace.shutdown};

  auto const dispatcher_stats = comm_dispatcher.stats();

  auto const recv_stats = data_channel->statistics();
  auto const comm_stats = comm_channel->statistics();

  trace.trace().add_time("Tcomm.enqueue", comm_stats.enqueue_time);
  trace.trace().add_time("Tcomm.dequeue", comm_stats.dequeue_time);

  trace.trace().add_time("Tcomp.enqueue", recv_stats.enqueue_time);
  trace.trace().add_time("Tcomp.dequeue", recv_stats.dequeue_time);

  trace.trace().add_time("Tcomm.dispatch", dispatcher_stats.dispatch_time);
  trace.trace().add_time("Tcomm.progress", dispatcher_stats.completion_time);
  trace.trace().add_time("Tcomm.callback", dispatcher_stats.callback_time);
  trace.trace().add_time("Tcomm.total", dispatcher_stats.total_time);

  trace.trace().put(
      "comm.iterations", static_cast<int>(dispatcher_stats.iterations));
}
}  // namespace fmpi
#endif
