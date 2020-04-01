#ifndef FMPI_ALLTOALL_WAITALL_HPP
#define FMPI_ALLTOALL_WAITALL_HPP

#include <sstream>

#include <fmpi/Debug.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/mpi/Request.hpp>

#include <rtlx/Trace.hpp>

#include <tlx/math/div_ceil.hpp>
#include <tlx/simple_vector.hpp>

namespace fmpi {

namespace detail {
template <class T>
struct SlidingReqWindow;
}  // namespace detail

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs = 1>
inline void RingWaitall(
    InputIt               begin,
    OutputIt              out,
    int                   blocksize,
    ::mpi::Context const& ctx,
    Op&&                  op) {
  using value_type = typename std::iterator_traits<OutputIt>::value_type;
  using buffer_t   = ::tlx::
      SimpleVector<value_type, ::tlx::SimpleVectorMode::NoInitNoDestroy>;

  auto me = ctx.rank();
  auto nr = ctx.size();

  std::ostringstream os;
  os << Schedule::NAME << "Waitall" << NReqs;
  auto trace = rtlx::Trace{os.str()};

  auto const schedule = Schedule{};

  FMPI_DBG_STREAM(
      "running algorithm " << os.str() << ", blocksize: " << blocksize);

  if (nr < 3) {
    detail::Ring_lt3(
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

  trace.put(N_COMM_ROUNDS, static_cast<int>(nrounds));

  FMPI_DBG_RANGE(out, out + nr * blocksize);
}
namespace detail {

template <class T>
struct SlidingReqWindow {
 private:
  /// nongrowing vector without initialization
  using simple_vector =
      tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

  using iterator  = typename simple_vector::iterator;
  using iter_pair = std::pair<iterator, iterator>;

  simple_vector storage_{};

 public:
  SlidingReqWindow(std::size_t winsize, std::size_t blocksize)
    : storage_(2 * winsize * blocksize)
    , winsize_(winsize)
    , recvbuf_(storage_.begin())
    , mergebuf_(std::next(recvbuf_, storage_.size() / 2)) {
    // We explitly reserve one additional chunk for the local portion
    pending_.reserve(winsize + 1);
    ready_.reserve(winsize + 1);
  }

  [[nodiscard]] auto winsize() const noexcept {
    return winsize_;
  }

  void buffer_swap() {
    // swap chunks
    std::swap(pending_, ready_);
    // swap buffers
    std::swap(recvbuf_, mergebuf_);
  }

  [[nodiscard]] auto rbuf() const noexcept -> iterator {
    return recvbuf_;
  }

  [[nodiscard]] auto mergebuf() const noexcept -> iterator {
    return mergebuf_;
  }

  auto& pending_pieces() noexcept {
    return pending_;
  }

  [[nodiscard]] auto const& pending_pieces() const noexcept {
    return pending_;
  }

  auto& ready_pieces() noexcept {
    return ready_;
  }

  [[nodiscard]] auto const& ready_pieces() const noexcept {
    return ready_;
  }

  std::size_t            winsize_{};
  iterator               recvbuf_{};
  iterator               mergebuf_{};
  std::vector<iter_pair> pending_{};
  std::vector<iter_pair> ready_{};
};
}  // namespace detail
}  // namespace fmpi
#endif
