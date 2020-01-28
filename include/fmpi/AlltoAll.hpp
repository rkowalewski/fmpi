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

template <typename F, typename... Ts>
inline auto make_async(F&& f, Ts&&... params) {
  // Suggested in effective modern C++ to get true asynchrony
  return std::async(
      std::launch::async, std::forward<F>(f), std::forward<Ts>(params)...);
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
    rtlx::TimeTrace&    trace) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;

  using merge_buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto chunks = std::vector<std::pair<InputIt, InputIt>>{};
  chunks.reserve(2);

  auto const me = ctx.rank();

  chunks.emplace_back(
      std::make_pair(begin + me * blocksize, begin + (me + 1) * blocksize));

  if (ctx.size() == 1) {
    trace.tick(MERGE);
    op(chunks, out);
    trace.tock(MERGE);
    return;
  }

  auto other = static_cast<mpi::Rank>(1 - me);

  trace.tick(COMMUNICATION);
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
  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

  {
    chunks.emplace_back(std::make_pair(
        out + other * blocksize, out + (other + 1) * blocksize));
    merge_buffer_t buffer{ctx.size() * blocksize};
    op(chunks, buffer.begin());

    std::move(buffer.begin(), buffer.end(), out);
  }

  trace.tock(MERGE);

  trace.put(detail::N_COMM_ROUNDS, 1);
}

template <class T, std::size_t Capacity>
using lfq_fifo =
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>>;

template <class Iterator, std::size_t Capacity>
void push_fifo(
    Iterator begin,
    Iterator end,
    lfq_fifo<typename std::iterator_traits<Iterator>::value_type, Capacity>&
        fifo) {
  FMPI_DBG_STREAM(
      "pushing " << std::distance(begin, end) << " on fifo of capacity "
                 << Capacity);
  FMPI_ASSERT(
      std::distance(begin, end) <=
      static_cast<typename std::iterator_traits<Iterator>::difference_type>(
          Capacity));
  for (auto it = begin; it != end;) {
    it = fifo.push(it, end);
  }
}

template <class TaskQueue, class ReadyQueue, class OutputIterator, class Op>
inline OutputIterator compute(
    TaskQueue&                     q_tasks,
    ReadyQueue&                    q_done,
    typename TaskQueue::value_type local_task,
    std::size_t                    n_remote_tasks,
    std::size_t                    n_producers,
    OutputIterator                 output,
    Op&&                           op,
    std::size_t                    op_threshold) {
  using value_type =
      typename std::iterator_traits<OutputIterator>::value_type;
  using Task = typename TaskQueue::value_type;
  // chunks to merge
  std::vector<Task> chunks_to_merge;
  chunks_to_merge.reserve(n_remote_tasks);
  chunks_to_merge.push_back(local_task);

  auto const blocksize = std::distance(local_task.first, local_task.second);

  // prefix sum over all processed chunks
  std::vector<Task> processed;
  processed.reserve(n_producers);

  std::size_t n_arrivals = 0;

  auto first = output;

  FMPI_DBG(n_remote_tasks);

  while (n_arrivals < n_remote_tasks) {
    auto const n_old = chunks_to_merge.size();
    auto const n_new = q_tasks.pop(std::back_inserter(chunks_to_merge));
    FMPI_DBG(n_new);
    auto const n_merges = n_old + n_new;

    n_arrivals += n_new;
    FMPI_DBG(n_arrivals);

    // minimum number of chunks to merge: ideally we have a full level2
    // cache
    bool const enough_work = n_merges >= op_threshold;

    FMPI_DBG(n_merges);

    if (enough_work) {
      // 2) merge all chunks
      auto last = op(chunks_to_merge, first);

      // 4) release completed buffers for future receives
      auto fbuf = std::begin(chunks_to_merge);

      if (first == output) {
        std::advance(fbuf, 1);
      }

      auto const nready = std::distance(fbuf, std::end(chunks_to_merge));

      FMPI_DBG(nready);

      detail::push_fifo(fbuf, std::end(chunks_to_merge), q_done);

      chunks_to_merge.clear();

      // 3) increase out iterator
      processed.emplace_back(std::make_pair(first, last));
      std::swap(first, last);
    } else {
      // we can eventually replace this with a wait method
      // std::this_thread::yield();
    }
  }

  auto const nels = static_cast<std::size_t>(n_producers) * blocksize;

  using merge_buffer_t =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto mergeBuffer = merge_buffer_t{nels};
  // generate pairs of chunks to merge
  std::copy(
      std::begin(processed),
      std::end(processed),
      std::back_inserter(chunks_to_merge));

  FMPI_DBG(chunks_to_merge.size());
  // merge
  auto const last = op(chunks_to_merge, mergeBuffer.begin());

  FMPI_ASSERT(last == mergeBuffer.end());

  FMPI_DBG(processed);

  return std::move(mergeBuffer.begin(), mergeBuffer.end(), output);
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

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();
  auto const me = ctx.rank();

  std::ostringstream os;
  os << Schedule::NAME << "Waitsome" << NReqs;

  auto trace = rtlx::TimeTrace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  trace.tick(COMMUNICATION);

  constexpr auto winreqs = 2 * NReqs;

  auto const totalExchanges = static_cast<size_t>(nr - 1);
  auto const reqsInFlight   = std::min(totalExchanges, NReqs);

  using window_buffer =
      tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto winbuf = window_buffer{blocksize * 2 * reqsInFlight};
  using bufit = typename window_buffer::iterator;
  using chunk = std::pair<bufit, bufit>;

  auto lfq_done = detail::lfq_fifo<chunk, NReqs>{};

  auto lfq_freelist = detail::lfq_fifo<chunk, winreqs>{};
  {
    auto chunks = std::array<chunk, winreqs>{};
    auto r      = range<std::size_t>(chunks.size());

    std::transform(
        std::begin(r),
        std::end(r),
        std::begin(chunks),
        [begin = std::begin(winbuf), blocksize](auto idx) {
          return std::make_pair(
              std::next(begin, idx * blocksize),
              std::next(begin, (idx + 1) * blocksize));
        });

    detail::push_fifo(std::begin(chunks), std::end(chunks), lfq_freelist);
  }

  auto async_compute =
      detail::make_async([&lfq_done,
                          &lfq_freelist,
                          lchunk = std::make_pair(
                              std::next(begin, me * blocksize),
                              std::next(begin, (me + 1) * blocksize)),
                          totalExchanges,
                          nr,
                          out,
                          op] {
        return detail::compute(
            lfq_done,
            lfq_freelist,
            lchunk,
            totalExchanges,
            nr,
            out,
            op,
            utilization_threshold);
      });

  StackVector<MPI_Request, winreqs> reqs{};
  FMPI_ASSERT(reqs->capacity() == winreqs);
  reqs->resize(reqs->capacity(), MPI_REQUEST_NULL);

  StackVector<chunk, NReqs> arrived_chunks;
  arrived_chunks->resize(arrived_chunks->capacity());

  StackVector<int, winreqs> indices{};
  FMPI_ASSERT(indices->capacity() == winreqs);
  indices->resize(indices->capacity());
  // initially we can use the full array of request indices for send and
  // receives
  std::iota(
      indices->begin(), std::next(indices->begin(), reqsInFlight * 2), 0);

  StackVector<chunk, NReqs> occupied{};
  occupied->resize(occupied->capacity());

  FMPI_ASSERT(2 * reqsInFlight <= reqs->size());

  auto rschedule = [&ctx](auto phase) {
    Schedule commAlgo{};
    return commAlgo.recvRank(ctx, phase);
  };

  auto rbufAlloc = [&occupied, &lfq_freelist](auto /*peer*/, auto reqIdx) {
    chunk c;
    while (!lfq_freelist.pop(c)) {
      // std::this_thread::yield();
    }
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

  // Yes, this is an int due to API requirements in our trace module
  int n_comm_rounds = 0;

  // Indices smaller than the pivot are receive requests, then we have send
  // requests
  auto sreqs_pivot = std::next(indices->begin(), reqsInFlight);

  do {
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

    {
      FMPI_DBG_STREAM("pushing " << nrecv << " on comp queue...");

      // ensure that we never try to push more than we have available capacity
      FMPI_ASSERT(nrecv <= arrived_chunks->size());

      auto last = std::transform(
          reqsCompleted.first,
          sreqs_pivot,
          arrived_chunks->begin(),
          [&occupied](auto reqIdx) { return occupied[reqIdx]; });

      detail::push_fifo(arrived_chunks->begin(), last, lfq_done);
    }

    nSlotsRecv = std::min<std::size_t>(totalExchanges - nrreqs, nrecv);
    nSlotsSend = std::min<std::size_t>(totalExchanges - nsreqs, nsent);

    FMPI_ASSERT((reqsCompleted.first + nSlotsRecv) <= sreqs_pivot);

  } while (nc_reqs < total_reqs);

  trace.tock(COMMUNICATION);

  trace.put(detail::N_COMM_ROUNDS, n_comm_rounds);

  FMPI_DBG("waiting for compute");

  trace.tick(MERGE);
  auto const last = async_compute.get();
  trace.tock(MERGE);

  FMPI_ASSERT(last == out + (std::size_t(nr) * blocksize));
}  // namespace fmpi

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

  auto trace = rtlx::TimeTrace{os.str()};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::scatteredPairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  trace.tick(COMMUNICATION);

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
  using chunk      = std::pair<mpi::Rank, Span<value_type>>;
  auto ready_tasks = buffered_channel<chunk>{n_rounds};

  std::vector<std::pair<Ticket, Span<value_type>>> blocks{};
  blocks.reserve(reqsInFlight);

  // Dispatcher Thread
  fmpi::CommDispatcher dispatcher{winsz};
  dispatcher.start_worker();
  dispatcher.pinToCore(config.dispatcher_core);

  auto fut_comm = fmpi::async<void>(config.scheduler_core, [&]() {
    auto enqueue_alloc =
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
            [](MPI_Status, Ticket) {
              std::cout << "callback fire for send\n";
            });
        FMPI_DBG(sticket.id);
      }
    }
  });

  using iterator = OutputIt;

  auto f_comp = fmpi::async<iterator>(config.comp_core, [&]() -> iterator {
    // chunks to merge
    std::vector<std::pair<OutputIt, OutputIt>> chunks_to_merge;
    chunks_to_merge.reserve(ctx.size());
    // local task
    chunks_to_merge.push_back(std::make_pair(
        std::next(begin, ctx.rank() * blocksize),
        std::next(begin, (ctx.rank() + 1) * blocksize)));

    // prefix sum over all processed chunks
    auto n = ctx.size() - 1;

    std::vector<std::pair<OutputIt, OutputIt>> processed;
    processed.reserve(n);

    // minimum number of chunks to merge: ideally we have a full level2
    // cache

    constexpr auto op_threshold = (NReqs / 2);

    auto dst = out;

    while ((n--) != 0u) {
      FMPI_DBG(buf_alloc.allocatedBlocks());
      FMPI_DBG(buf_alloc.allocatedHeapBlocks());
      FMPI_DBG(buf_alloc.isFull());
      auto ready = ready_tasks.value_pop();

      auto const [peer, s] = ready;

      FMPI_ASSERT(s.size() == std::size_t(blocksize));

      chunks_to_merge.emplace_back(s.data(), s.data() + s.size());

      bool const enough_work = chunks_to_merge.size() >= op_threshold;

      if (enough_work) {
        // merge all chunks
        auto last = op(chunks_to_merge, dst);

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

    auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

    using merge_buffer_t =
        tlx::SimpleVector<value_type, tlx::SimpleVectorMode::NoInitNoDestroy>;

    auto mergeBuffer = merge_buffer_t{nels};

    // generate pairs of chunks to merge
    std::copy(
        std::begin(processed),
        std::end(processed),
        std::back_inserter(chunks_to_merge));

    FMPI_DBG(chunks_to_merge.size());
    // merge
    auto const last = op(chunks_to_merge, mergeBuffer.begin());

    for (auto f = std::begin(chunks_to_merge);
         f != std::prev(std::end(chunks_to_merge), processed.size());
         ++f) {
      FMPI_DBG_STREAM("release p " << f->first);
      buf_alloc.dispose(f->first);
    }

    FMPI_ASSERT(last == mergeBuffer.end());

    FMPI_DBG(processed);

    return std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  });


  iterator ret;
  try {
    fut_comm.wait();
    trace.tock(COMMUNICATION);
    trace.tick(MERGE);
    ret = f_comp.get();
    trace.tock(MERGE);
  } catch (...) {
    throw std::runtime_error("asynchronous Alltoall failed");
  }

  dispatcher.loop_until_done();

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
  auto trace = rtlx::TimeTrace{os.str()};

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

  trace.tick(COMMUNICATION);

  std::tie(rphase, sphase) =
      moveReqWindow(std::make_pair(rphase, sphase), reqWin);

  FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));

  reqWin.buffer_swap();

  FMPI_ASSERT(reqWin.pending_pieces().empty());
  FMPI_ASSERT(!reqWin.ready_pieces().empty());

  trace.tock(COMMUNICATION);

  for (auto&& win : range<std::size_t>(nrounds - 1)) {
    static_cast<void>(win);

    trace.tick(COMMUNICATION);
    std::tie(rphase, sphase) =
        moveReqWindow(std::make_pair(rphase, sphase), reqWin);
    trace.tock(COMMUNICATION);

    trace.tick(MERGE);
    op(reqWin.ready_pieces(), mergebuf);
    auto const nMerged = reqWin.ready_pieces().size() * blocksize;
    mergebuf += nMerged;
    mergePsum.push_back(mergePsum.back() + nMerged);
    reqWin.ready_pieces().clear();
    trace.tock(MERGE);

    trace.tick(COMMUNICATION);
    FMPI_CHECK_MPI(mpi::waitall(&(*std::begin(reqs)), &(*std::end(reqs))));
    reqWin.buffer_swap();
    trace.tock(COMMUNICATION);

    FMPI_DBG_RANGE(out, mergebuf);
  }

  trace.tick(MERGE);

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

  trace.tock(MERGE);

  trace.put(detail::N_COMM_ROUNDS, nrounds);

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

  auto trace = rtlx::TimeTrace{"AlltoAll"};

  trace.tick(COMMUNICATION);

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  FMPI_CHECK_MPI(mpi::alltoall(
      std::addressof(*begin), blocksize, &rbuf[0], blocksize, ctx));

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

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

  trace.tock(MERGE);
}
#if 0

template <
    class Schedule,
    bool isBlocking,
    class InputIt,
    class OutputIt,
    class Op>
inline void scatteredPairwise(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op)
{
  auto nr = ctx.size();
  auto me = ctx.rank();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto rbuf = std::unique_ptr<value_type[]>(new value_type[nr * blocksize]);

  std::ostringstream os;
  os << Schedule::NAME;
  if (isBlocking) {
    os << "Blocking";
  }

  auto trace = rtlx::TimeTrace{os.str()};

  trace.tick(COMMUNICATION);
  std::copy(
      begin + me * blocksize,
      begin + me * blocksize + blocksize,
      &rbuf[0] + me * blocksize);

  auto commAlgo = Schedule{};

  std::size_t              nreqs = isBlocking ? 2 : nr * 2;
  std::vector<MPI_Request> reqs(nreqs, MPI_REQUEST_NULL);

  for (int r = 0; r < static_cast<int>(nr); ++r) {
    auto sendto   = commAlgo.sendRank(ctx, r);
    auto recvfrom = commAlgo.recvRank(ctx, r);

    if (sendto == me) {
      sendto = mpi::Rank{};
    }
    if (recvfrom == me) {
      recvfrom = mpi::Rank{};
    }
    if (sendto == MPI_PROC_NULL && recvfrom == MPI_PROC_NULL) {
      continue;
    }

    FMPI_DBG(sendto);
    FMPI_DBG(recvfrom);

    FMPI_CHECK(mpi::irecv(
        std::next(&(rbuf[0]), recvfrom * blocksize),
        blocksize,
        recvfrom,
        EXCH_TAG_RING,
        ctx,
        &reqs[isBlocking ? 0 : r]));

    FMPI_CHECK(mpi::isend(
        std::next(begin, sendto * blocksize),
        blocksize,
        sendto,
        EXCH_TAG_RING,
        ctx,
        &reqs[isBlocking ? 1 : nr + r]));

    if (isBlocking) {
      mpi::waitall(&(*reqs.begin()), &(*reqs.end()));
    }
  }

  if (!isBlocking) {
    mpi::waitall(&(*reqs.begin()), &(*reqs.end()));
  }

  trace.tock(COMMUNICATION);

  trace.tick(MERGE);

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

  trace.tock(MERGE);
}
#endif
}  // namespace fmpi
#endif
