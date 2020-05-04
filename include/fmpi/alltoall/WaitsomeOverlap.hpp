#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Config.hpp>
#include <fmpi/Dispatcher.hpp>
#include <fmpi/allocator/HeapAllocator.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/container/StackContainer.hpp>
#include <fmpi/container/buffered_channel.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
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
  duration main{0};

  TimeTrace(std::string name)
    : name_(std::move(name))
    , trace_(name_) {
  }

  TimeTrace(const TimeTrace& other) = delete;
  TimeTrace(TimeTrace&& other)      = delete;
  TimeTrace& operator=(const TimeTrace& other) = delete;
  TimeTrace& operator=(TimeTrace&& other) = delete;

  ~TimeTrace() {
    trace_.add_time("Tmain.initialize", initialize);
    trace_.add_time("Tmain.dispatch", dispatch);
    trace_.add_time("Tmain.receive", receive);
    trace_.add_time("Tmain.shutdown", shutdown);
    trace_.add_time(kComputationTime, compute);
    trace_.add_time("Tmain.idle", idle);
    trace_.add_time("Tmain.main", main);
  }

  std::string_view name() const noexcept {
    return name_;
  }

  rtlx::Trace& trace() noexcept {
    return trace_;
  }
};

template <class T, class Allocator>
class Piece {
  using range = gsl::span<T>;
  range      span_{};
  Allocator* alloc_{};

 public:
  using value_type     = T;
  using size_type      = typename range::size_type;
  using iterator       = T*;
  using const_iterator = T const*;

  constexpr Piece() = default;

  constexpr Piece(gsl::span<T> span) noexcept
    : Piece(span, nullptr) {
  }

  constexpr Piece(gsl::span<T> span, Allocator* alloc) noexcept
    : span_(span)
    , alloc_(alloc) {
    FMPI_DBG(span.size());
  }

  ~Piece() {
    if (alloc_) {
      alloc_->deallocate(span_.data(), span_.size());
    }
  }

  Piece(Piece const&) = delete;

  constexpr Piece(Piece&& other) noexcept {
    *this = std::move(other);
  }

  Piece& operator=(Piece const&) = delete;

  constexpr Piece& operator=(Piece&& other) noexcept {
    if (this == &other) return *this;

    using std::swap;
    swap(span_, other.span_);
    swap(alloc_, other.alloc_);

    // reset other to null span to avoid double frees
    other.span_  = gsl::span<T>{};
    other.alloc_ = nullptr;

    return *this;
  }

  constexpr iterator data() noexcept {
    return span_.data();
  }

  constexpr const_iterator data() const noexcept {
    return span_.data();
  }

  //! return number of items in range
  constexpr size_type size() const noexcept {
    return span_.size();
  }

  //! return mutable T* to first element
  constexpr iterator begin() noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  constexpr const_iterator begin() const noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  constexpr const_iterator cbegin() const noexcept {
    return begin();
  }

  //! return mutable T* beyond last element
  constexpr iterator end() noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  constexpr const_iterator end() const noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  constexpr const_iterator cend() const noexcept {
    return end();
  }
};

template <class T>
using simple_vector =
    tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

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

  auto tt = detail::TimeTrace<timer>{std::string{Schedule::NAME} +
                                     std::string{algorithm_name} +
                                     std::to_string(NReqs)};

  FMPI_DBG_STREAM(
      "running algorithm " << tt.name() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), tt.trace());
    return;
  }

  auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

  timer t_init{tt.initialize};
  timer t_main{tt.main};

  Schedule const commAlgo{};

  // Each round is composed of an isend-irecv pair...
  constexpr std::size_t messages_per_round = 2;

  auto const n_exchanges  = ctx.size() - 1;
  auto const n_messages   = n_exchanges * messages_per_round;
  auto const n_rounds     = commAlgo.phaseCount(ctx);
  auto const reqsInFlight = std::min(std::size_t(n_rounds), NReqs);
  auto const winsz        = reqsInFlight * messages_per_round;

  // intermediate buffer for two pipelines
  using thread_alloc = ThreadAllocator<value_type>;

  std::size_t const nthreads = config.num_threads;

  FMPI_DBG(nthreads);
  FMPI_DBG(blocksize);

  auto buf_alloc = thread_alloc{};

  // queue for ready tasks
  using chunk = std::pair<mpi::Rank, gsl::span<value_type>>;

#if 0
  //auto data_channel = boost::lockfree::spsc_queue<chunk>{n_rounds};
#else
  using channel_t = SPSCNChannel<chunk>;
#endif

  using dispatcher_t = CommDispatcher<mpi::testsome>;

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
        auto allocated_span = gsl::span(buffer, blocksize);
        FMPI_DBG(allocated_span.data());

        // add the buffer to the message
        message.set_buffer(allocated_span);

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
    timer t_dispatch{tt.dispatch};

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
    t_dispatch.finish();
  }

  {
    timer t_comp{tt.compute};
    FMPI_DBG("processing message arrivals...");

    // chunks to merge
    // std::vector<chunk>                         chunks;
    using iter_pair = std::pair<OutputIt, OutputIt>;

    auto enough_work = [](auto /*c_first*/, auto /*c_last*/) -> bool {
      // minimum number of chunks to merge: ideally we have a full level2
      // cache
      // constexpr auto op_threshold = (NReqs / 2);
      // return static_cast<std::size_t>(std::distance(c_first, c_last)) > 1
      //        op_threshold;
      return true;
    };

    // prefix sum over all processed chunks
    auto       d_first = out;
    auto const d_last  = std::next(out, nels);

    using piece        = detail::Piece<value_type, thread_alloc>;
    using merge_buffer = detail::simple_vector<value_type>;

    using merge_chunk = std::variant<piece, merge_buffer>;

    std::vector<merge_chunk> pieces;
    pieces.reserve(ctx.size());

    {
      timer t_receive{tt.receive};
      auto  local_span =
          gsl::span{std::next(begin, ctx.rank() * blocksize),
                    std::next(begin, (ctx.rank() + 1) * blocksize)};

      pieces.emplace_back(local_span);

      chunk task{};
      while (data_channel->wait_dequeue(task)) {
        auto [source, span] = task;
        pieces.emplace_back(piece{span, &buf_alloc});

        if (enough_work(pieces.begin(), pieces.end())) {
          // merge all chunks
          std::vector<iter_pair> chunks;
          chunks.reserve(pieces.size());

          std::transform(
              std::begin(pieces),
              std::end(pieces),
              std::back_inserter(chunks),
              [](auto& piece) {
                return std::visit(
                    [](auto&& v) -> iter_pair {
                      FMPI_ASSERT(v.begin() <= v.end());
                      return std::make_pair(v.begin(), v.end());
                    },
                    piece);
              });

          auto const n_elements = std::accumulate(
              std::begin(chunks),
              std::end(chunks),
              std::size_t(0),
              [](auto acc, auto const& piece) {
                return acc + std::distance(piece.first, piece.second);
              });

          FMPI_DBG(n_elements);

          auto last = d_last;

          using diff_t =
              typename std::iterator_traits<OutputIt>::difference_type;

          if (diff_t(n_elements) <= std::distance(d_first, d_last)) {
            last = op(chunks, d_first);
            pieces.emplace_back(piece{gsl::span{d_first, last}});
            std::swap(d_first, last);
          } else {
            auto buffer = merge_buffer{n_elements};
            last        = op(chunks, buffer.begin());

            FMPI_ASSERT(last == buffer.end());

            pieces.emplace_back(std::move(buffer));
          }

          pieces.erase(std::begin(pieces), std::prev(std::end(pieces)));
        }
      }
      t_receive.finish();
    }

    {
      FMPI_DBG("final merge");
      FMPI_DBG(pieces.size());

      std::vector<iter_pair> chunks;
      chunks.reserve(pieces.size());

      std::transform(
          std::begin(pieces),
          std::end(pieces),
          std::back_inserter(chunks),
          [](auto& piece) {
            return std::visit(
                [](auto&& v) -> iter_pair {
                  return std::make_pair(v.begin(), v.end());
                },
                piece);
          });

      auto const n_elements = std::accumulate(
          std::begin(chunks),
          std::end(chunks),
          std::size_t(0),
          [](auto acc, auto const& piece) {
            return acc + std::distance(piece.first, piece.second);
          });

      FMPI_DBG(n_elements);
      FMPI_ASSERT(n_elements == nels);

      auto       mergeBuffer = merge_buffer{nels};
      auto const last        = op(chunks, mergeBuffer.begin());

      FMPI_ASSERT(last == mergeBuffer.end());

      std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
    }
    t_comp.finish();
  }

  t_main.finish();

  {
    timer t_idle{tt.idle};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    comm_dispatcher.stop_worker();
  }

  timer t_shutdown{tt.shutdown};

  auto const dispatcher_stats = comm_dispatcher.stats();

  auto const recv_stats = data_channel->statistics();
  auto const comm_stats = comm_channel->statistics();

  tt.trace().add_time("Tcomm.enqueue", comm_stats.enqueue_time);
  tt.trace().add_time("Tcomm.dequeue", comm_stats.dequeue_time);

  tt.trace().add_time("Tcomp.enqueue", recv_stats.enqueue_time);
  tt.trace().add_time("Tcomp.dequeue", recv_stats.dequeue_time);

  tt.trace().add_time("Tcomm.dispatch", dispatcher_stats.dispatch_time);
  tt.trace().add_time("Tcomm.progress", dispatcher_stats.progress_time);
  tt.trace().add_time("Tcomm.completion", dispatcher_stats.completion_time);
  tt.trace().add_time("Tcomm.callback", dispatcher_stats.callback_time);
  tt.trace().add_time("Tcomm.total", dispatcher_stats.total_time);
  tt.trace().add_time("Tcomm.queue_time", dispatcher_stats.queue_time);
  tt.trace().add_time("Tcomm.life_time", dispatcher_stats.life_time);
  tt.trace().add_time("Tcomm.thread_time", dispatcher_stats.thread_time);
  tt.trace().put("Tcomm.high_watermark", dispatcher_stats.high_watermark);
  tt.trace().put("Tcomm.nreqs_completion", dispatcher_stats.nreqs_completion);

  tt.trace().put(kCommRounds, dispatcher_stats.iterations);

}  // namespace fmpi
}  // namespace fmpi
#endif
