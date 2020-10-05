#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Pinning.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/concurrency/BufferedChannel.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/util/Trace.hpp>
#include <string_view>
#include <utility>

namespace fmpi {

namespace detail {

using namespace std::literals::string_view_literals;
constexpr auto initialize = "Tmain.initialize"sv;
constexpr auto dispatch   = "Tmain.dispatch"sv;
constexpr auto receive    = "Tmain.receive"sv;
constexpr auto shutdown   = "Tmain.shutdown"sv;
constexpr auto idle       = "Tmain.idle"sv;
constexpr auto compute    = kComputationTime;

template <class T, class Allocator>
class Piece;

template <class T>
using simple_vector =
    tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

template <class Schedule>
class CollectiveCtx {
 public:
  CollectiveCtx(
      const void*         sendbuf,
      std::size_t         sendcount,
      MPI_Datatype        sendtype,
      void*               recvbuf,
      std::size_t         recvcount,
      MPI_Datatype        recvtype,
      mpi::Context const& ctx)
    : sendbuf_(sendbuf)
    , sendcount_(sendcount)
    , sendtype_(sendtype)
    , recvbuf_(recvbuf)
    , recvcount_(recvcount)
    , recvtype_(recvtype)
    , comm_(ctx.mpiComm())
    , schedule_(ctx) {
  }

  collective_future execute();

 private:
  const void*  sendbuf_;
  std::size_t  sendcount_;
  MPI_Datatype sendtype_;
  void*        recvbuf_;
  std::size_t  recvcount_;
  MPI_Datatype recvtype_;
  MPI_Comm     comm_;
  Schedule     schedule_;
};  // namespace detail

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
  constexpr auto algorithm_name = std::string_view("WaitsomeOverlap");

  auto const& config = Pinning::instance();

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto const nr = ctx.size();

  auto const name = std::string{Schedule::NAME} +
                    std::string{algorithm_name} + std::to_string(NReqs);

  auto trace = MultiTrace{std::string_view(name)};

  FMPI_DBG_STREAM(
      "running algorithm " << trace.name() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::ring_pairwise_lt3(
        begin, out, blocksize, ctx, std::forward<Op&&>(op), trace);
    return;
  }

  auto const nels = static_cast<std::size_t>(ctx.size()) * blocksize;

  steady_timer t_init{trace.duration(detail::initialize)};

  auto const commAlgo = Schedule{ctx};

  // Each round is composed of an isend-irecv pair...
  // constexpr std::size_t messages_per_round = 2;

  // exclude the local message to MPI_Self
  auto const n_exchanges  = ctx.size() - 1;
  auto const n_rounds     = commAlgo.phaseCount();
  auto const reqsInFlight = std::min(std::size_t(n_rounds), NReqs);
  // auto const winsz        = reqsInFlight * messages_per_round;

  // intermediate buffer for two pipelines
  using thread_alloc = ThreadAllocator<value_type>;

  std::size_t const nthreads = config.num_threads;

  FMPI_DBG(nthreads);
  FMPI_DBG(blocksize);

  auto buf_alloc = thread_alloc{};

  // queue for ready tasks
  // using segment = std::pair<mpi::Rank, gsl::span<value_type>>;

  // using channel_t = SPSCNChannel<segment>;

  // auto data_channel = std::make_shared<channel_t>(n_exchanges);

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(reqsInFlight);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto queue   = future.arrival_queue();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

  schedule_state->register_signal(
      message_type::IRECV, [buf_alloc, blocksize](Message& message) mutable {
        // allocator some buffer
        auto* buffer = buf_alloc.allocate(blocksize);
        FMPI_ASSERT(buffer);
        auto allocated_span = gsl::span(buffer, blocksize);
        FMPI_DBG(allocated_span.data());

        // add the buffer to the message
        message.set_buffer(allocated_span);
      });

  schedule_state->register_callback(
      message_type::IRECV, [sptr = queue](std::vector<Message> msgs) mutable {
        FMPI_ASSERT(sptr);
        std::move(
            std::begin(msgs), std::end(msgs), std::back_inserter(*sptr));
      });

  auto& dispatcher = dispatcher_executor();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  // auto future = collective_future{std::move(schedule_state), queue};

  t_init.finish();

  {
    FMPI_DBG("Sending essages");
    steady_timer t_dispatch{trace.duration(detail::dispatch)};

    for (auto&& r : range(commAlgo.phaseCount())) {
      auto const rpeer = commAlgo.recvRank(r);
      auto const speer = commAlgo.sendRank(r);

      if (rpeer != ctx.rank()) {
        auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::IRECV, recv);
      }

      if (speer != ctx.rank()) {
        auto span = gsl::span(std::next(begin, speer * blocksize), blocksize);

        auto send = Message{span, speer, kTagRing, ctx.mpiComm()};

        dispatcher.schedule(hdl, message_type::ISEND, send);
      }
    }

    dispatcher.commit(hdl);
  }

  using iter_pair = std::pair<OutputIt, OutputIt>;
  using piece     = detail::Piece<value_type, thread_alloc>;
  using chunk     = std::variant<piece, detail::simple_vector<value_type>>;
  using pieces_t  = std::vector<chunk>;

  pieces_t pieces;
  pieces.reserve(ctx.size());

  {
    using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;

    steady_timer t_receive{trace.duration(detail::receive)};

    FMPI_DBG("processing message arrivals...");

    // prefix sum over all processed chunks
    auto       d_first = out;
    auto const d_last  = std::next(out, nels);

    auto enough_work = [&config](
                           typename pieces_t::const_iterator c_first,
                           typename pieces_t::const_iterator c_last) -> bool {
      auto const     npieces    = std::distance(c_first, c_last);
      constexpr auto min_pieces = 1;

      auto const nbytes = std::accumulate(
          c_first, c_last, std::size_t(0), [](auto acc, auto const& c) {
            return acc + std::visit(
                             [](auto&& v) -> std::size_t {
                               return v.size() * sizeof(value_type);
                             },
                             c);
          });

      auto const ncpus_rank = std::size_t(config.domain_size);

      return npieces > min_pieces && (nbytes >= (kCacheSizeL2 * ncpus_rank));
    };

    {
      auto local_span = gsl::span{
          std::next(begin, ctx.rank() * blocksize),
          std::next(begin, (ctx.rank() + 1) * blocksize)};

      pieces.emplace_back(piece{local_span});

      // segment task{};
      std::size_t n = n_exchanges;
      // while (data_channel->wait_dequeue(task)) {
      std::vector<Message> msgs;
      msgs.reserve(n);
      while (n) {
        std::size_t m = 0;
        queue->pop_all(std::back_inserter(msgs), m);

        std::transform(
            std::begin(msgs),
            std::end(msgs),
            std::back_inserter(pieces),
            [palloc = &buf_alloc](auto& msg) {
              auto span = gsl::span(
                  static_cast<value_type*>(msg.writable_buffer()),
                  msg.count());

              FMPI_DBG_STREAM(
                  "receiving segment: " << std::make_pair(
                      msg.peer(), span.data()));

              return piece{span, palloc};
            });

        msgs.clear();
        n -= m;

        if (enough_work(pieces.begin(), pieces.end())) {
          steady_timer t_comp{trace.duration(kComputationTime)};
          // we temporarily pause t_receive and run t_comp.
          scoped_timer_switch switcher{t_receive, t_comp};
          // merge all chunks
          std::vector<iter_pair> chunks;
          chunks.reserve(pieces.size());

          std::transform(
              std::begin(pieces),
              std::end(pieces),
              std::back_inserter(chunks),
              [](auto& c) {
                return std::visit(
                    [](auto&& v) -> iter_pair {
                      FMPI_ASSERT(v.begin() <= v.end());
                      return std::make_pair(v.begin(), v.end());
                    },
                    c);
              });

          auto const n_elements = std::accumulate(
              std::begin(chunks),
              std::end(chunks),
              std::size_t(0),
              [](auto acc, auto const& c) {
                return acc + std::distance(c.first, c.second);
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
            auto buffer = detail::simple_vector<value_type>{n_elements};
            last        = op(chunks, buffer.begin());

            FMPI_ASSERT(last == buffer.end());

            pieces.emplace_back(std::move(buffer));
          }

          FMPI_DBG_STREAM("clearing " << pieces.size() << "pieces");
          pieces.erase(std::begin(pieces), std::prev(std::end(pieces)));
        }
      }
    }
  }

  {
    steady_timer t_comp{trace.duration(kComputationTime)};
    FMPI_DBG("final merge");
    FMPI_DBG(pieces.size());

    std::vector<iter_pair> chunks;
    chunks.reserve(pieces.size());

    std::transform(
        std::begin(pieces),
        std::end(pieces),
        std::back_inserter(chunks),
        [](auto& c) {
          return std::visit(
              [](auto&& v) -> iter_pair {
                return std::make_pair(v.begin(), v.end());
              },
              c);
        });

    auto const n_elements = std::accumulate(
        std::begin(chunks),
        std::end(chunks),
        std::size_t(0),
        [](auto acc, auto const& c) {
          return acc + std::distance(c.first, c.second);
        });

    FMPI_DBG(n_elements);
    FMPI_ASSERT(n_elements == nels);

    auto       mergeBuffer = detail::simple_vector<value_type>{nels};
    auto const last        = op(chunks, mergeBuffer.begin());

    FMPI_ASSERT(last == mergeBuffer.end());

    std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  }

  {
    steady_timer t_idle{trace.duration(detail::idle)};
    // We definitely have to wait here because although all data has arrived
    // there might still be pending tasks for other peers (e.g. sends)
    // comm_dispatcher.loop_until_done();
    auto ret = future.get();
    FMPI_ASSERT(ret == MPI_SUCCESS);
  }

  // steady_timer t_comp{trace.duration(detail::shutdown)};

  // auto const& dispatcher_stats = comm_dispatcher.stats();
  // trace.merge(dispatcher_stats.begin(), dispatcher_stats.end());

  // auto const recv_stats = data_channel->statistics();
  // auto const comm_stats = comm_channel->statistics();

  // trace.duration("Tcomm.enqueue") = comm_stats.enqueue_time;
  // trace.duration("Tcomm.dequeue") = comm_stats.dequeue_time;
  // trace.duration("Tcomp.enqueue") = recv_stats.enqueue_time;
  // trace.duration("Tcomp.dequeue") = recv_stats.dequeue_time;
}

namespace detail {

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

  constexpr explicit Piece(gsl::span<T> span) noexcept
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
    if (this == &other) {
      return *this;
    }

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

  [[nodiscard]] constexpr const_iterator data() const noexcept {
    return span_.data();
  }

  //! return number of items in range
  [[nodiscard]] constexpr size_type size() const noexcept {
    return span_.size();
  }

  //! return mutable T* to first element
  constexpr iterator begin() noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return span_.data();
  }
  //! return constant T* to first element
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
    return begin();
  }

  //! return mutable T* beyond last element
  constexpr iterator end() noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return data() + size();
  }
  //! return constant T* beyond last element
  [[nodiscard]] constexpr const_iterator cend() const noexcept {
    return end();
  }
};
}  // namespace detail
}  // namespace fmpi
#endif
