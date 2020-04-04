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

namespace fmpi {
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
      nthreads * kMaxContiguousBufferSize / sizeof(value_type) * n_pipelines;

  auto const n_buffer_nels = std::min<std::size_t>(
      std::min(required, capacity),
      std::numeric_limits<typename buffer_allocator::index_type>::max());

  FMPI_ASSERT(required <= capacity);

  auto buf_alloc = buffer_allocator{
      static_cast<typename buffer_allocator::index_type>(n_buffer_nels)};

  FMPI_DBG(n_buffer_nels);

  // queue for ready tasks
  using chunk = std::pair<mpi::Rank, Span<value_type>>;
#if 0
  auto ready_tasks = boost::lockfree::spsc_queue<chunk>{n_rounds};
#else
  auto ready_tasks = buffered_channel<chunk>{n_rounds};
#endif

  std::vector<std::pair<Ticket, Span<value_type>>> blocks{};
  blocks.reserve(reqsInFlight * n_pipelines);

  // Dispatcher Thread

  fmpi::CommDispatcher<mpi::testsome> dispatcher{winsz};
  dispatcher.start_worker();
  dispatcher.pinToCore(config.dispatcher_core);

  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;

  duration t_schedule{};

  auto fut_comm = fmpi::async(
      config.scheduler_core,
      [&buf_alloc,
       &dispatcher,
       &blocks,
       &ready_tasks,
       &t_schedule,
       &ctx,       // const ref
       blocksize,  // const
       commAlgo,   // const
       begin,      // const
       n_rounds    // const
  ]() {
        timer t{t_schedule};
        auto  enqueue_alloc =
            [&buf_alloc, &blocks, blocksize](fmpi::Ticket ticket) {
              // allocator some buffer
              auto* b = buf_alloc.allocate(blocksize);
              FMPI_DBG_STREAM("allocate p " << b);
              auto s = fmpi::make_span(b, blocksize);
              auto r = std::make_pair(ticket, s);
              FMPI_DBG(r);
              blocks.push_back(std::move(r));
              return s;
            };

        auto dequeue = [&blocks, &ready_tasks](
                           fmpi::Ticket ticket, mpi::Rank peer) {
          FMPI_DBG(ticket);
          auto it = std::find_if(
              std::begin(blocks), std::end(blocks), [ticket](const auto& v) {
                return v.first == ticket;
              });

          FMPI_ASSERT(it != std::end(blocks));

          ready_tasks.push(std::make_pair(peer, it->second));
          blocks.erase(it);
        };

        for (auto&& r : fmpi::range(n_rounds)) {
          auto const rpeer = commAlgo.recvRank(ctx, r);
          auto const speer = commAlgo.sendRank(ctx, r);

          if (rpeer != ctx.rank()) {
            auto const rticket = dispatcher.postAsync(
                request_type::IRECV,
                [cb = std::move(enqueue_alloc), rpeer, &ctx](
                    MPI_Request* req, fmpi::Ticket ticket) -> int {
                  auto s = cb(ticket);

                  FMPI_DBG_STREAM(
                      "mpi::irecv from rank " << rpeer << ", span: " << s);

                  return mpi::irecv(
                      s.data(), s.size(), rpeer, EXCH_TAG_RING, ctx, req);
                },
                [cb = std::move(dequeue), rpeer](
                    MPI_Status status, Ticket ticket) {
                  FMPI_ASSERT(status.MPI_ERROR == MPI_SUCCESS);
                  cb(ticket, rpeer);
                });

            FMPI_DBG(rticket.id);
          }

          if (speer != ctx.rank()) {
            auto sb = Span<const value_type>(
                std::next(begin, speer * blocksize), blocksize);
            auto const sticket = dispatcher.postAsync(
                request_type::ISEND,
                [&ctx, speer, sb](MPI_Request* req, Ticket) -> int {
                  FMPI_DBG_STREAM(
                      "mpi::isend to rank " << speer << ", span: " << sb);

                  return mpi::isend(
                      sb.data(),
                      sb.size(),
                      static_cast<mpi::Rank>(speer),
                      EXCH_TAG_RING,
                      ctx,
                      req);
                },
                Function<void(MPI_Status, Ticket)>{});
            FMPI_DBG(sticket.id);
          }
        }
      });

  using iterator = OutputIt;

  struct ComputeTime {
    duration queue{0};
    duration comp{0};
    duration total{0};
  } t_compute;

  auto f_comp = fmpi::async(
      config.comp_core,
      [&ready_tasks,
       &buf_alloc,
       &ctx,
       &t_compute,
       blocksize,
       begin,
       out,
       comp = std::forward<Op>(op)]() -> iterator {
        timer t{t_compute.total};
        // chunks to merge
        std::vector<std::pair<OutputIt, OutputIt>> chunks;
        std::vector<std::pair<OutputIt, OutputIt>> processed;

        chunks.reserve(ctx.size());
        processed.reserve(ctx.size());
        // local task
        chunks.push_back(std::make_pair(
            std::next(begin, ctx.rank() * blocksize),
            std::next(begin, (ctx.rank() + 1) * blocksize)));

        // minimum number of chunks to merge: ideally we have a full level2
        // cache

        constexpr auto op_threshold = (NReqs / 2);

        // prefix sum over all processed chunks
        auto ntasks  = ctx.size() - 1;
        auto d_first = out;
        while (ntasks != 0u) {
          {
            timer t{t_compute.queue};

            FMPI_DBG(buf_alloc.allocatedBlocks());
            FMPI_DBG(buf_alloc.allocatedHeapBlocks());
            FMPI_DBG(buf_alloc.isFull());

#if 0
              auto const n =
                  ready_tasks.consume_all([&chunks](auto const& v) {
                    auto const [peer, s] = v;
                    chunks.emplace_back(s.data(), s.data() + s.size());
                  });

              ntasks -= n;
#else
            auto c = ready_tasks.value_pop();
            chunks.emplace_back(
                c.second.data(), c.second.data() + c.second.size());
            ntasks--;
#endif
          }

          auto const enough_work = chunks.size() >= op_threshold;

          if (enough_work) {
            timer t{t_compute.comp};
            // merge all chunks
            auto d_last = comp(chunks, d_first);

            for (auto f =
                     std::next(std::begin(chunks), processed.size() == 0);
                 f != std::end(chunks);
                 ++f) {
              FMPI_DBG_STREAM("release p " << f->first);
              buf_alloc.dispose(f->first);
            }

            chunks.clear();

            // 3) increase out iterator
            processed.emplace_back(d_first, d_last);
            std::swap(d_first, d_last);
          } else {
            // TODO: merge processed chunks
          }
        }

        {
          timer      t{t_compute.comp};
          auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

          using merge_buffer_t = tlx::SimpleVector<
              value_type,
              tlx::SimpleVectorMode::NoInitNoDestroy>;

          auto mergeBuffer = merge_buffer_t{nels};

          FMPI_DBG(chunks.size());

          // generate pairs of chunks to merge
          std::copy(
              std::begin(processed),
              std::end(processed),
              std::back_inserter(chunks));

          FMPI_DBG(chunks.size());

          auto const last = comp(chunks, mergeBuffer.begin());

          for (auto f = std::next(std::begin(chunks), processed.size() == 0);
               f != std::prev(std::end(chunks), processed.size());
               ++f) {
            FMPI_DBG_STREAM("release p " << f->first);
            buf_alloc.dispose(f->first);
          }

          FMPI_ASSERT(last == mergeBuffer.end());

          FMPI_DBG(processed);

          return std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
        }
      });

  iterator ret;
  try {
    fut_comm.wait();
    t_comm.finish();
    {
      rtlx::TimeTrace tt(trace, COMPUTATION);
      ret = f_comp.get();
    }
  } catch (...) {
    throw std::runtime_error("asynchronous Alltoall failed");
  }

#ifndef NDEBUG
  dispatcher.loop_until_done();
  auto const stats = dispatcher.stats();

  FMPI_ASSERT(stats.ntasks == 0);

  trace.add_time("DispatcherThread.dispatch_time", stats.dispatch_time);
  trace.add_time("DispatcherThread.queue_time", stats.queue_time);
  trace.add_time("DispatcherThread.completion_time", stats.completion_time);
  trace.put(
      "DispatcherThread.iterations", static_cast<int>(stats.iterations));
#endif

  trace.add_time("ComputeThread.queue_time", t_compute.queue);
  trace.add_time("ComputeThread.compute_time", t_compute.comp);
  trace.add_time("ComputeThread.total_time", t_compute.total);
  trace.add_time("ScheduleThread", t_schedule);

  FMPI_ASSERT(ret == out + ctx.size() * blocksize);
}
}  // namespace fmpi
#endif
