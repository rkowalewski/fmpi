#ifndef ALGORITHMS_MERGE_HPP
#define ALGORITHMS_MERGE_HPP

#include <Params.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/concurrency/Future.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/util/NumericRange.hpp>
#include <fmpi/util/Trace.hpp>
#include <numeric>
#include <string_view>
#include <tlx/algorithm.hpp>
#include <tlx/simple_vector.hpp>
#include <utility>

namespace benchmark {

using vector_times =
    std::vector<std::pair<std::string, std::chrono::nanoseconds>>;

namespace detail {

using namespace std::literals::string_view_literals;
constexpr auto idle    = "Tcomm.idle"sv;
constexpr auto receive = "Tcomm.receive"sv;
constexpr auto merge   = "Tmerge"sv;

template <class S, class R>
vector_times merge_pieces(
    TypedCollectiveArgs<S, R> const& collective_args,
    fmpi::collective_future          future,
    R*                               out);
}  // namespace detail

template <class RandomAccessIterator1, class RandomAccessIterator2>
RandomAccessIterator2 parallel_merge(
    std::vector<std::pair<RandomAccessIterator1, RandomAccessIterator1>> seqs,
    RandomAccessIterator2                                                res,
    std::size_t                                                          n) {
  // parallel merge does not support inplace merging
  // n must be the number of elements in all sequences
  assert(!seqs.empty());
  assert(res);

#if 0
  return tlx::parallel_multiway_merge(
      std::begin(seqs),
      std::end(seqs),
      res,
      n,
      std::less<>{},
      tlx::MultiwayMergeAlgorithm::MWMA_ALGORITHM_DEFAULT,
      tlx::MultiwayMergeSplittingAlgorithm::MWMSA_DEFAULT,
      omp_get_max_threads());
#else
  return tlx::multiway_merge(std::begin(seqs), std::end(seqs), res, n);
#endif
}

template <class S, class R>
vector_times merge_async(
    TypedCollectiveArgs<S, R> const& collective_args,
    fmpi::collective_future          future,
    R*                               out) {
  using scoped_timer = rtlx::steady_timer;
  using duration     = scoped_timer::duration;

  // if (1) {
  if (future.is_deferred() || future.is_ready()) {
    vector_times times;
    times.emplace_back(detail::idle, duration{});
    times.emplace_back(detail::merge, duration{});

    auto&        d_idle  = times[0].second;
    auto&        d_merge = times[1].second;
    scoped_timer t_merge{d_merge};

    auto const nr        = collective_args.comm.size();
    auto const blocksize = static_cast<uint32_t>(collective_args.recvcount);
    std::vector<std::pair<R*, R*>> chunks;
    chunks.reserve(nr);

    auto range = fmpi::range<uint32_t>(0, nr * blocksize, blocksize);

    auto const& config = fmpi::Pinning::instance();

    FMPI_ASSERT(
        config.num_threads == static_cast<uint32_t>(omp_get_max_threads()));

    std::transform(
        std::begin(range),
        std::end(range),
        std::back_inserter(chunks),
        [buf = static_cast<R*>(collective_args.recvbuf),
         blocksize](auto const& offset) {
          auto f = std::next(buf, offset);
          auto l = std::next(f, blocksize);
          return std::make_pair(f, l);
        });

    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(chunks.begin(), chunks.end(), g);

    {
      using scoped_timer_switch = rtlx::ScopedTimerSwitch<scoped_timer>;
      scoped_timer t_idle{d_idle};
      // pause compute
      scoped_timer_switch switcher{t_merge, t_idle};
      future.wait();
    }

    // FMPI_DBG(rtlx::to_seconds(d_idle.second));

    parallel_merge(chunks, out, nr * blocksize);
    return times;
  }
  return detail::merge_pieces(collective_args, std::move(future), out);
}

namespace detail {

template <class T, class Allocator>
class Piece;

template <class T>
using simple_vector =
    tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

template <class S, class R>
vector_times merge_pieces(
    TypedCollectiveArgs<S, R> const& collective_args,
    fmpi::collective_future          future,
    R*                               out) {
  using value_type          = R;
  using steady_timer        = rtlx::steady_timer;
  using duration            = steady_timer::duration;
  using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;
  // using thread_alloc        = fmpi::ThreadAllocator<value_type>;
  using iter_pair = std::pair<value_type*, value_type*>;
  // using piece               = Piece<value_type, thread_alloc>;
  // using chunk               = std::variant<piece,
  // simple_vector<value_type>>; using pieces_t            =
  // std::vector<chunk>;

  using namespace std::chrono_literals;

  vector_times times;
  times.emplace_back(detail::idle, duration{});
  times.emplace_back(detail::receive, duration{});
  times.emplace_back(detail::merge, duration{});
  auto& d_idle    = times[0].second;
  auto& d_receive = times[1].second;
  auto& d_merge   = times[2].second;

  steady_timer t_idle{d_idle};

  auto const& ctx         = collective_args.comm;
  auto const  blocksize   = collective_args.recvcount;
  auto const  nels        = ctx.size() * blocksize;
  std::size_t n_exchanges = ctx.size();
  auto        queue       = future.arrival_queue();
  // auto        buf_alloc   = thread_alloc{};

  // auto const* begin = static_cast<value_type
  // const*>(collective_args.sendbuf);
  // auto* out = static_cast<value_type*>(collective_args.recvbuf);

  std::vector<iter_pair> chunks;
  chunks.reserve(n_exchanges);

  // auto       d_first = out;
  auto const d_last = std::next(out, nels);
  {
    steady_timer        t_receive{d_receive};
    scoped_timer_switch switcher{t_idle, t_receive};
    while (n_exchanges--) {
      fmpi::Message msg{};
      queue->pop(msg);
      FMPI_ASSERT(msg.recvcount() == blocksize);
      auto* first = static_cast<value_type*>(msg.recvbuffer());
      auto* last  = std::next(first, blocksize);
      chunks.emplace_back(first, last);
    }
  }

  FMPI_ASSERT(t_idle.running());

  {
    steady_timer        t_merge{d_merge};
    scoped_timer_switch switcher{t_idle, t_merge};
    auto                last = parallel_merge(chunks, out, nels);

    FMPI_ASSERT(last == d_last);
  }

  return times;
}

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

#if 0
template <class S, class R>
vector_times merge_pieces(
    TypedCollectiveArgs<S, R> const& collective_args,
    fmpi::collective_future          future,
    R*                               out) {
  using value_type          = R;
  using steady_timer        = rtlx::steady_timer;
  using duration            = steady_timer::duration;
  using scoped_timer_switch = rtlx::ScopedTimerSwitch<steady_timer>;
  using thread_alloc        = fmpi::ThreadAllocator<value_type>;
  using iter_pair           = std::pair<value_type*, value_type*>;
  using piece               = Piece<value_type, thread_alloc>;
  using chunk               = std::variant<piece, simple_vector<value_type>>;
  using pieces_t            = std::vector<chunk>;

  vector_times times;
  times.emplace_back(detail::t_receive, duration{});
  times.emplace_back(detail::t_merge, duration{});
  auto& d_receive = times[0].second;
  auto& d_merge   = times[1].second;

  steady_timer t_merge{d_merge};

  auto const& ctx         = collective_args.comm;
  auto const  blocksize   = collective_args.recvcount;
  auto const  nels        = ctx.size() * blocksize;
  std::size_t n_exchanges = ctx.size();
  auto        queue       = future.arrival_queue();
  // auto        buf_alloc   = thread_alloc{};

  // auto const* begin = static_cast<value_type
  // const*>(collective_args.sendbuf);
  // auto* out = static_cast<value_type*>(collective_args.recvbuf);

  pieces_t               pieces;
  std::vector<iter_pair> chunks;
  pieces.reserve(ctx.size());
  chunks.reserve(n_exchanges);

  auto       d_first = out;
  auto const d_last  = std::next(out, nels);
  {
    steady_timer        t_receive{d_receive};
    scoped_timer_switch switcher{t_merge, t_receive};

    // prefix sum over all processed chunks

    auto enough_work = [](std::size_t n) -> bool {
      auto const nbytes     = n * sizeof(value_type);
      auto&      config     = fmpi::Pinning::instance();
      auto const ncpus_rank = std::size_t(config.domain_size);

      return nbytes >= (fmpi::kCacheSizeL2 * ncpus_rank);
    };

    std::vector<fmpi::Message> msgs;
    msgs.reserve(n_exchanges);
    while (n_exchanges) {
      std::size_t m = 0;
      queue->pop_all(std::back_inserter(msgs), m);

      std::transform(
          std::begin(msgs),
          std::end(msgs),
          std::back_inserter(pieces),
          [/*palloc = &buf_alloc*/](auto& msg) {
            auto span = gsl::span(
                static_cast<value_type*>(msg.recvbuffer()), msg.recvcount());

#if 0
            FMPI_DBG_STREAM(
                "receiving segment: " << std::make_pair(
                    msg.peer(), span.data()));
#endif

            // return piece{span, palloc};
            return piece{span};
          });

      msgs.clear();
      n_exchanges -= m;

      auto const n_elements = std::accumulate(
          std::begin(pieces),
          std::end(pieces),
          std::size_t(0),
          [](auto const& acc, auto const& c) {
            auto const n = std::visit(
                [](auto&& v) -> std::size_t { return v.size(); }, c);
            return acc + n;
          });

      constexpr auto min_pieces = std::size_t(1);
      if (pieces.size() > min_pieces and enough_work(n_elements)) {
        // we temporarily pause t_receive and run t_comp.
        scoped_timer_switch switcher{t_receive, t_merge};
        // merge all chunks

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

        FMPI_DBG(n_elements);

        auto last = d_last;

        using diff_t = typename std::iterator_traits<R*>::difference_type;

        if (diff_t(n_elements) <= std::distance(d_first, d_last)) {
          last = parallel_merge(chunks, d_first, n_elements);
          pieces.emplace_back(piece{gsl::span{d_first, last}});
          std::swap(d_first, last);
        } else {
          auto buffer = simple_vector<value_type>{n_elements};
          last        = parallel_merge(chunks, buffer.begin(), n_elements);

          FMPI_ASSERT(last == buffer.end());

          pieces.emplace_back(std::move(buffer));
        }

        FMPI_DBG_STREAM("clearing " << pieces.size() << " pieces");
        pieces.erase(std::begin(pieces), std::prev(std::end(pieces)));
        chunks.clear();
      }
      FMPI_ASSERT(t_receive.running());
      FMPI_ASSERT(not t_merge.running());
    }
  }

  FMPI_ASSERT(t_merge.running());
  FMPI_ASSERT(chunks.empty());

  FMPI_DBG(pieces.size());

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
      [](auto const& acc, auto const& c) {
        return acc + std::distance(c.first, c.second);
      });

  FMPI_DBG(n_elements);
  FMPI_ASSERT(n_elements == nels);

  auto* last = out;

  if (d_first != out) {
    // we used the out buffer alredy for temporary merges
    auto mergeBuffer = simple_vector<value_type>{nels};
    parallel_merge(chunks, mergeBuffer.begin(), nels);
    last = std::move(mergeBuffer.begin(), mergeBuffer.end(), out);
  } else {
    last = parallel_merge(chunks, out, nels);
  }

  FMPI_ASSERT(last == d_last);
  return times;
}
#endif

}  // namespace detail

}  // namespace benchmark

#endif
