#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <algorithm>
#include <cmath>
#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/container/StackContainer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/detail/Async.hpp>
#include <fmpi/detail/CommState.hpp>
#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Dispatcher.hpp>
#include <fmpi/mpi/Request.hpp>
#include <future>
#include <memory>
#include <numeric>
#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>
#include <stack>
#include <string_view>
#include <tlx/math/div_ceil.hpp>
#include <tlx/simple_vector.hpp>
#include <tlx/stack_allocator.hpp>
#include <utility>

// Other AllToAll Algorithms

namespace fmpi {

namespace detail {

template <class T, std::size_t N>
constexpr std::size_t maximum_stack_capacity() {
  return std::min(fmpi::kContainerStackSize / sizeof(T), N);
}

static constexpr char N_COMM_ROUNDS[] = "Ncomm_rounds";

template <class Schedule, class ReqIdx, class BufAlloc, class CommOp>
inline auto enqueueMpiOps(
    uint32_t   phase,
    mpi::Rank  me,
    uint32_t   reqsInFlight,
    Schedule&& partner,
    ReqIdx&&   reqIdx,
    BufAlloc&& bufAlloc,
    CommOp&&   commOp) {
  uint32_t nreqs;

  for (nreqs = 0; nreqs < reqsInFlight; ++phase) {
    auto peer = partner(phase);

    if (peer == me) {
      FMPI_DBG_STREAM("skipping local phase " << phase);
      continue;
    }

    auto idx = reqIdx(nreqs);

    FMPI_DBG_STREAM(
        "exchanging data with " << peer << " phase " << phase << " reqIdx "
                                << idx);

    auto buf = bufAlloc(peer, idx);

    FMPI_CHECK_MPI(commOp(buf, peer, idx));

    ++nreqs;
  }

  return phase;
}

template <class InputIt, class OutputIt, class Op>
inline void scatteredPairwise_lt3(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op,
    rtlx::Trace&        trace) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;

  using merge_buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto chunks = std::vector<std::pair<InputIt, InputIt>>{};
  chunks.reserve(2);

  auto const me = ctx.rank();

  chunks.emplace_back(
      std::make_pair(begin + me * blocksize, begin + (me + 1) * blocksize));

  if (ctx.size() == 1) {
    rtlx::TimeTrace tt(trace, COMPUTATION);
    op(chunks, out);
    return;
  }

  auto other = static_cast<mpi::Rank>(1 - me);

  {
    rtlx::TimeTrace tt(trace, COMMUNICATION);
    FMPI_CHECK_MPI(mpi::sendrecv(
        begin + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        out + other * blocksize,
        blocksize,
        other,
        EXCH_TAG_BRUCK,
        ctx));
  }

  {
    rtlx::TimeTrace tt(trace, COMPUTATION);

    chunks.emplace_back(std::make_pair(
        out + other * blocksize, out + (other + 1) * blocksize));
    merge_buffer_t buffer{ctx.size() * blocksize};
    op(chunks, buffer.begin());

    std::move(buffer.begin(), buffer.end(), out);
  }

  trace.put(detail::N_COMM_ROUNDS, 1);
}

template <class T, std::size_t Capacity>
using lfq_fifo =
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>>;

template <
    class InputIteratorIterator,
    class OutputIterator,
    class Op,
    class Pred>
std::pair<InputIteratorIterator, OutputIterator> apply_compute(
    InputIteratorIterator first,
    InputIteratorIterator last,
    OutputIterator        d_first,
    Op&&                  op,
    Pred&&                has_enough_work) {
  auto const n_merges = std::distance(first, last);

  if (!has_enough_work(first, last)) return std::make_pair(first, d_first);

  std::vector<
      typename std::iterator_traits<InputIteratorIterator>::value_type>
      chunks;

  chunks.reserve(n_merges);
  std::copy(first, last, std::back_inserter(chunks));

  // 2) merge all chunks
  auto d_last = op(chunks, d_first);

  return std::make_pair(last, d_last);
}
}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void scatteredPairwiseWaitsome(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  // Tuning Parameters:

  // NReqs: Maximum Number of pending receives

  // Defines the size of a task. Here we say that at least 50% of pending
  // receives need to be ready before we combine them in a local computation.
  // This of course depends on the size of a pending receive.
  //
  // An alternative may be: Given that messages are relatively small we can
  // wait until either the L2 or L3 cache is full to maximize the benefit for
  // local computation
  constexpr auto utilization_threshold = (NReqs / 2);
  static_assert(
      utilization_threshold > 1, "at least two concurrent receives required");

  auto has_enough_work = [](auto first_piece, auto last_piece) {
    auto const n = std::distance(first_piece, last_piece);
    FMPI_ASSERT(n >= 0);

    return std::size_t(n) >= utilization_threshold;
  };

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::ostringstream os;
  os << Schedule::NAME << "Waitsome" << NReqs;

  auto trace = rtlx::Trace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  rtlx::TimeTrace t_prepare{trace, COMMUNICATION};

  constexpr auto winreqs = 2 * NReqs;

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  using window_buffer =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto winbuf = window_buffer{blocksize * 2 * reqsInFlight};
  using bufit = typename window_buffer::iterator;
  using piece = std::pair<bufit, bufit>;

  auto lfq_done = detail::lfq_fifo<piece, NReqs>{};

  using freelist_t = StackContainer<
      tlx::RingBuffer<piece, tlx::StackAllocator<piece, kContainerStackSize>>,
      kContainerStackSize>;

  freelist_t freelist{};
  freelist->allocate(winreqs);
  FMPI_ASSERT(freelist->size() == 0);

  {
    auto r = range<std::size_t>(winreqs);

    std::transform(
        std::begin(r),
        std::end(r),
        std::front_inserter(freelist.container()),
        [begin = std::begin(winbuf), blocksize](auto idx) {
          return std::make_pair(
              std::next(begin, idx * blocksize),
              std::next(begin, (idx + 1) * blocksize));
        });
  }

  StackVector<MPI_Request, winreqs> reqs{};
  reqs->resize(winreqs, MPI_REQUEST_NULL);
  FMPI_ASSERT(reqs->capacity() == winreqs);

  // StackVector<piece, NReqs> arrived_chunks;
  // arrived_chunks->resize(NReqs);
  std::vector<piece> arrived_chunks;
  arrived_chunks.reserve(NReqs + 1);

  arrived_chunks.push_back(std::make_pair(
      std::next(begin, me * blocksize),
      std::next(begin, (me + 1) * blocksize)));

  std::vector<piece> pieces_done;
  pieces_done.reserve(nr);

  StackVector<int, winreqs> indices{};
  indices->resize(winreqs);
  FMPI_ASSERT(indices->capacity() == winreqs);
  // initially we can use the full array of request indices for send and
  // receives
  std::iota(
      indices->begin(), std::next(indices->begin(), reqsInFlight * 2), 0);

  StackVector<piece, NReqs> occupied{};
  occupied->resize(NReqs);

  FMPI_ASSERT(2 * reqsInFlight <= reqs->size());

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&occupied, &freelist](auto /*peer*/, auto reqIdx) {
    auto c = freelist->back();
    freelist->pop_back();
    return (occupied[reqIdx] = c);
  };

  auto receiveOp = [&reqs, &ctx](auto chunk, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("receiving from " << peer << " reqIdx " << reqIdx);

    auto const nels = std::distance(chunk.first, chunk.second);
    return mpi::irecv(
        &*chunk.first, nels, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  auto sschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.sendRank(ctx, phase);
  };

  auto sbufAlloc = [begin, blocksize](auto peer, auto /*reqIdx*/) {
    auto first = std::next(begin, peer * blocksize);
    return std::make_pair(first, std::next(first, blocksize));
  };

  auto sendOp = [&reqs, &ctx](auto chunk, auto peer, auto reqIdx) {
    FMPI_DBG_STREAM("sending to " << peer << " reqIdx " << reqIdx);

    auto const nels = std::distance(chunk.first, chunk.second);
    return mpi::isend(
        &*chunk.first, nels, peer, EXCH_TAG_RING, ctx, &reqs[reqIdx]);
  };

  std::size_t const total_reqs = 2 * totalExchanges;
  std::size_t       nc_reqs    = 0;
  std::size_t       nsreqs     = 0;
  std::size_t       nrreqs     = 0;
  std::size_t       sphase     = 0;
  std::size_t       rphase     = 0;
  std::size_t       nSlotsRecv = reqsInFlight;
  std::size_t       nSlotsSend = reqsInFlight;
  auto              d_first    = out;

  // Yes, this is an int due to API requirements in our trace module
  int n_comm_rounds = 0;

  // Indices smaller than the pivot are receive requests, then we have send
  // requests
  auto sreqs_pivot = std::next(indices->begin(), reqsInFlight);

  t_prepare.finish();

  do {
    rtlx::TimeTrace t_comm{trace, COMMUNICATION};
    ++n_comm_rounds;

    FMPI_DBG(n_comm_rounds);

    FMPI_DBG("receiving...");
    FMPI_DBG(nSlotsRecv);

    rphase = detail::enqueueMpiOps(
        rphase,
        me,
        nSlotsRecv,
        rschedule,
        [it = indices->begin()](auto nreqs) { return *std::next(it, nreqs); },
        rbufAlloc,
        receiveOp);

    nrreqs += nSlotsRecv;

    FMPI_DBG("sending...");
    FMPI_DBG(nSlotsSend);

    sphase = detail::enqueueMpiOps(
        sphase,
        me,
        nSlotsSend,
        sschedule,
        [it = sreqs_pivot](auto nreqs) { return *std::next(it, nreqs); },
        sbufAlloc,
        sendOp);

    nsreqs += nSlotsSend;

    FMPI_DBG(nc_reqs);

    auto const null_reqs = static_cast<std::size_t>(
        std::count(reqs->begin(), reqs->end(), MPI_REQUEST_NULL));

    auto const n_active_reqs = reqs->size() - null_reqs;
    FMPI_DBG(n_active_reqs);

    // (nc_reqs < total_reqs) $\implies$ (there is at least one non-null)
    FMPI_ASSERT((!(nc_reqs < total_reqs)) || null_reqs < reqs->size());

    int* lastIdx{};
    FMPI_CHECK_MPI(mpi::testsome(
        &(*reqs->begin()),
        &(*reqs->end()),
        &(*indices->begin()),
        MPI_STATUSES_IGNORE,
        lastIdx));

    FMPI_ASSERT(lastIdx >= &*indices->begin());

    auto const nCompleted = std::distance(&(*indices->begin()), lastIdx);

    FMPI_DBG(nCompleted);

    nc_reqs += nCompleted;

    auto const reqsCompleted =
        std::make_pair(indices->begin(), indices->begin() + nCompleted);

    for (auto it = reqsCompleted.first; it < reqsCompleted.second; ++it) {
      reqs[*it] = MPI_REQUEST_NULL;
    }

    FMPI_DBG_RANGE(reqsCompleted.first, reqsCompleted.second);

    // we want all receive requests on the left, and send requests on the
    // right
    sreqs_pivot = std::partition(
        reqsCompleted.first, reqsCompleted.second, [reqsInFlight](auto req) {
          // left half of reqs array array are receives
          return req < static_cast<int>(reqsInFlight);
        });

    auto const nrecv = static_cast<std::size_t>(
        std::distance(reqsCompleted.first, sreqs_pivot));

    FMPI_DBG(nrecv);

    auto const nsent = nCompleted - nrecv;

    FMPI_DBG(nsent);

    nSlotsRecv = std::min<std::size_t>(totalExchanges - nrreqs, nrecv);
    nSlotsSend = std::min<std::size_t>(totalExchanges - nsreqs, nsent);

    FMPI_ASSERT((reqsCompleted.first + nSlotsRecv) <= sreqs_pivot);

    t_comm.finish();

    {
      rtlx::TimeTrace t_comp{trace, COMPUTATION};
      FMPI_DBG_STREAM("pushing " << nrecv << " on comp queue...");

      // ensure that we never try to push more than we have available
      // capacity
      FMPI_ASSERT(nrecv <= arrived_chunks.capacity());

      std::transform(
          reqsCompleted.first,
          sreqs_pivot,
          std::back_inserter(arrived_chunks),
          [&occupied](auto reqIdx) { return occupied[reqIdx]; });

      FMPI_DBG(arrived_chunks.size());

      auto [last_piece, d_last] = detail::apply_compute(
          arrived_chunks.begin(),
          arrived_chunks.end(),
          d_first,
          std::forward<Op>(op),
          has_enough_work);

      FMPI_DBG_RANGE(d_first, d_last);
      auto const has_computed = d_first != d_last;
      if (has_computed) {
        auto first_to_release = arrived_chunks.begin();
        std::advance(first_to_release, (d_first == out));
        // release chunks
        std::copy(
            first_to_release,
            last_piece,
            std::front_inserter(freelist.container()));

        pieces_done.emplace_back(d_first, d_last);
        std::swap(d_first, d_last);

        arrived_chunks.erase(arrived_chunks.begin(), last_piece);
      }
    }
  } while (nc_reqs < total_reqs);

  trace.put(detail::N_COMM_ROUNDS, n_comm_rounds);

  {
    rtlx::TimeTrace t_comp{trace, COMPUTATION};
    auto const      nels = static_cast<std::size_t>(nr) * blocksize;

    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto mergeBuffer = merge_buffer_t{nels};

    std::copy(
        std::begin(pieces_done),
        std::end(pieces_done),
        std::back_inserter(arrived_chunks));

    FMPI_DBG(arrived_chunks.size());

    for (auto&& r : arrived_chunks) {
      FMPI_DBG_RANGE(r.first, r.second);
    }

    auto [last_piece, d_last] = detail::apply_compute(
        std::begin(arrived_chunks),
        std::end(arrived_chunks),
        mergeBuffer.begin(),
        std::forward<Op>(op),
        [](auto first, auto last) { return std::distance(first, last) > 0; });

    FMPI_DBG_RANGE(mergeBuffer.begin(), mergeBuffer.end());
    FMPI_DBG(std::make_pair(d_last, mergeBuffer.end()));
    FMPI_ASSERT(d_last == mergeBuffer.end());

    std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  }
}

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 2>
inline void scatteredPairwiseWaitsomeOverlap(
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
    detail::scatteredPairwise_lt3(
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

  uint16_t const buffersz  = winsz * blocksize * n_pipelines;
  auto           buf_alloc = buffer_allocator{buffersz};

  FMPI_DBG(buffersz);

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

  fmpi::CommDispatcher dispatcher{winsz};
  dispatcher.start_worker();
  dispatcher.pinToCore(config.dispatcher_core);

  using timer    = rtlx::Timer<>;
  using duration = typename timer::duration;

  duration t_schedule{};

  auto fut_comm = fmpi::async<void>(
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
  ]()
  {
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

  auto f_comp = fmpi::async<iterator>(
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
        std::vector<std::pair<OutputIt, OutputIt>> chunks_to_merge;
        chunks_to_merge.reserve(ctx.size());
        // local task
        chunks_to_merge.push_back(std::make_pair(
            std::next(begin, ctx.rank() * blocksize),
            std::next(begin, (ctx.rank() + 1) * blocksize)));

        // prefix sum over all processed chunks
        auto ntasks = ctx.size() - 1;

        std::vector<std::pair<OutputIt, OutputIt>> processed;
        processed.reserve(ntasks);

        // minimum number of chunks to merge: ideally we have a full level2
        // cache

        constexpr auto op_threshold = (NReqs / 2);

        auto dst = out;

        while (ntasks != 0u) {
          {
            timer t{t_compute.queue};

            FMPI_DBG(buf_alloc.allocatedBlocks());
            FMPI_DBG(buf_alloc.allocatedHeapBlocks());
            FMPI_DBG(buf_alloc.isFull());

#if 0
              auto const n =
                  ready_tasks.consume_all([&chunks_to_merge](auto const& v) {
                    auto const [peer, s] = v;
                    chunks_to_merge.emplace_back(s.data(), s.data() + s.size());
                  });

              ntasks -= n;
#else
            auto c = ready_tasks.value_pop();
            chunks_to_merge.emplace_back(
                c.second.data(), c.second.data() + c.second.size());
            ntasks--;
#endif
          }

          bool const enough_work = chunks_to_merge.size() >= op_threshold;

          if (enough_work) {
            timer t{t_compute.comp};
            // merge all chunks
            auto last = comp(chunks_to_merge, dst);

            auto f_buf = chunks_to_merge.begin();
            if (processed.size() == 0) {
              std::advance(f_buf, 1);
            }

            for (; f_buf != chunks_to_merge.end(); ++f_buf) {
              FMPI_DBG_STREAM("release p " << f_buf->first);
              buf_alloc.dispose(f_buf->first);
            }

            chunks_to_merge.clear();

            // 3) increase out iterator
            processed.emplace_back(std::make_pair(dst, last));
            std::swap(dst, last);
          }
        }

        {
          timer t{t_compute.comp};
          auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

          using merge_buffer_t = tlx::
              SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

          auto mergeBuffer = merge_buffer_t{nels};

          // generate pairs of chunks to merge
          std::copy(
              std::begin(processed),
              std::end(processed),
              std::back_inserter(chunks_to_merge));

          FMPI_DBG(chunks_to_merge.size());
          // merge
          auto const last = comp(chunks_to_merge, mergeBuffer.begin());

          for (auto f = std::begin(chunks_to_merge);
               f != std::prev(std::end(chunks_to_merge), processed.size());
               ++f) {
            FMPI_DBG_STREAM("release p " << f->first);
            buf_alloc.dispose(f->first);
          }

          FMPI_ASSERT(last == mergeBuffer.end());

          FMPI_DBG(processed);

          auto ret = std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
          return ret;
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

  auto const stats = dispatcher.stats();

  // dispatcher.loop_until_done();
  FMPI_ASSERT(stats.ntasks == 0);

  trace.add_time("ComputeThread.queue_time", t_compute.queue);
  trace.add_time("ComputeThread.compute_time", t_compute.comp);
  trace.add_time("ComputeThread.total_time", t_compute.total);
  trace.add_time("ScheduleThread", t_schedule);
  trace.add_time("DispatcherThread.dispatch_time", stats.dispatch_time);
  trace.add_time("DispatcherThread.queue_time", stats.queue_time);
  trace.add_time("DispatcherThread.completion_time", stats.completion_time);
  trace.put(
      "DispatcherThread.iterations", static_cast<int>(stats.iterations));

  FMPI_ASSERT(ret == out + ctx.size() * blocksize);
}

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 1>
inline void scatteredPairwiseWaitall(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;
  using buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto me = ctx.rank();
  auto nr = ctx.size();

  std::ostringstream os;
  os << Schedule::NAME << "Waitall" << NReqs;
  auto trace = rtlx::Trace{os.str()};

  auto const schedule = Schedule{};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  auto const totalExchanges = nr - 1;
  auto const reqsInFlight   = std::min<std::size_t>(totalExchanges, NReqs);

  auto reqWin = detail::SlidingReqWindow<value_type>{
      reqsInFlight, static_cast<std::size_t>(blocksize)};

  // We can save the local copy and do it with the merge operation itself
  reqWin.pending_pieces().push_back(std::make_pair(
      begin + me * blocksize, begin + me * blocksize + blocksize));

  // auto const phaseCount =
  // static_cast<std::size_t>(schedule.phaseCount(ctx));
  auto const nrounds = tlx::div_ceil(totalExchanges, reqsInFlight);

  std::vector<std::size_t> mergePsum;
  mergePsum.reserve(nrounds + 2);
  mergePsum.push_back(0);

  FMPI_DBG(nrounds);

  std::size_t rphase   = 0;
  std::size_t sphase   = 0;
  auto        mergebuf = out;

  auto const nels = nr * blocksize;
  buffer_t   buffer{nels};

  // auto const nbytes = nels * sizeof(value_type);
  // auto const allFitsInL2Cache = (nbytes < fmpi::CACHELEVEL2_SIZE);
  // constexpr bool allFitsInL2Cache = false;

  std::vector<MPI_Request> reqs(2 * reqsInFlight, MPI_REQUEST_NULL);

  auto moveReqWindow = [schedule, blocksize, &ctx, sbuffer = begin, &reqs](
                           auto initialPhases, auto& reqWin) {
    auto const phaseCount = schedule.phaseCount(ctx);
    auto const winsize    = reqWin.winsize();

    std::size_t rphase;
    std::size_t sphase;
    std::tie(rphase, sphase) = initialPhases;

    FMPI_DBG(initialPhases);

    for (std::size_t nrreqs = 0; nrreqs < winsize && rphase < phaseCount;
         ++rphase) {
      // receive from
      auto recvfrom = schedule.recvRank(ctx, rphase);

      if (recvfrom == ctx.rank()) {
        continue;
      }

      FMPI_DBG(recvfrom);

      auto* recvBuf = std::next(reqWin.rbuf(), nrreqs * blocksize);

      FMPI_CHECK_MPI(mpi::irecv(
          recvBuf, blocksize, recvfrom, EXCH_TAG_RING, ctx, &reqs[nrreqs]));

      reqWin.pending_pieces().push_back(
          std::make_pair(recvBuf, std::next(recvBuf, blocksize)));

      ++nrreqs;
    }

    for (std::size_t nsreqs = 0; nsreqs < winsize && sphase < phaseCount;
         ++sphase) {
      auto sendto = schedule.sendRank(ctx, sphase);

      if (sendto == ctx.rank()) {
        continue;
      }

      FMPI_DBG(sendto);

      FMPI_CHECK_MPI(mpi::isend(
          std::next(sbuffer, sendto * blocksize),
          blocksize,
          sendto,
          EXCH_TAG_RING,
          ctx,
          &reqs[winsize + nsreqs++]));
    }

    return std::make_pair(rphase, sphase);
  };

  {
    rtlx::TimeTrace{trace, COMMUNICATION};

    std::tie(rphase, sphase) =
        moveReqWindow(std::make_pair(rphase, sphase), reqWin);

    FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));

    reqWin.buffer_swap();

    FMPI_ASSERT(reqWin.pending_pieces().empty());
    FMPI_ASSERT(!reqWin.ready_pieces().empty());
  }

  for (auto&& win : range<std::size_t>(nrounds - 1)) {
    std::ignore = win;

    {
      rtlx::TimeTrace{trace, COMMUNICATION};
      std::tie(rphase, sphase) =
          moveReqWindow(std::make_pair(rphase, sphase), reqWin);
    }

    {
      rtlx::TimeTrace{trace, COMPUTATION};
      op(reqWin.ready_pieces(), mergebuf);
      auto const nMerged = reqWin.ready_pieces().size() * blocksize;
      mergebuf += nMerged;
      mergePsum.push_back(mergePsum.back() + nMerged);
      reqWin.ready_pieces().clear();
    }

    {
      rtlx::TimeTrace{trace, COMMUNICATION};
      FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));
      reqWin.buffer_swap();
    }

    FMPI_DBG_RANGE(out, mergebuf);
  }

  {
    rtlx::TimeTrace{trace, COMPUTATION};

    auto mergeSrc = &*out;
    auto target   = buffer.begin();

    // if we have intermediate merge operations then we have already merged
    // chunks in the out buffer and use the dedicated merge buffer as
    // destinated merged buffer. Otherwise, we merge directly into the out
    // buffer.
    FMPI_DBG(mergePsum);

    if (mergePsum.back() > 0) {
      FMPI_DBG(mergePsum);
      std::transform(
          std::begin(mergePsum),
          std::prev(std::end(mergePsum)),
          std::next(std::begin(mergePsum)),
          std::back_inserter(reqWin.ready_pieces()),
          [from = mergeSrc](auto first, auto last) {
            return std::make_pair(
                std::next(from, first), std::next(from, last));
          });
    }

    FMPI_DBG("final merge");
    FMPI_DBG(reqWin.ready_pieces().size());

    op(reqWin.ready_pieces(), target);

    FMPI_DBG(reqWin.ready_pieces());

    if (target != &*out) {
      std::move(target, target + nels, out);
    }
  }

  trace.put(detail::N_COMM_ROUNDS, static_cast<int>(nrounds));

  FMPI_DBG_RANGE(out, out + nr * blocksize);
}

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op) {
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto nr = ctx.size();

  auto trace = rtlx::Trace{"AlltoAll"};

  std::unique_ptr<value_type[]> rbuf;

  {
    rtlx::TimeTrace tt(trace, COMMUNICATION);

    rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

    FMPI_CHECK_MPI(mpi::alltoall(
        std::addressof(*begin), blocksize, &rbuf[0], blocksize, ctx));
  }

  {
    rtlx::TimeTrace tt(trace, COMPUTATION);

    std::vector<std::pair<InputIt, InputIt>> chunks;
    chunks.reserve(nr);

    auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

    std::transform(
        std::begin(range),
        std::end(range),
        std::back_inserter(chunks),
        [buf = rbuf.get(), blocksize](auto offset) {
          auto f = std::next(buf, offset);
          auto l = std::next(f, blocksize);
          return std::make_pair(f, l);
        });

    op(chunks, out);
  }
}
}  // namespace fmpi
#endif
