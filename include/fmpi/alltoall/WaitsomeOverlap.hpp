#ifndef FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP
#define FMPI_ALLTOALL_WAITSOMEOVERLAP_HPP

#include <fmpi/Debug.hpp>
#include <fmpi/Schedule.hpp>
#include <fmpi/alltoall/Detail.hpp>
#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/detail/CollectiveArgs.hpp>
#include <fmpi/memory/ThreadAllocator.hpp>
#include <fmpi/memory/detail/pointer_arithmetic.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <fmpi/util/Trace.hpp>
#include <numeric>
#include <string_view>
#include <tlx/math/div_ceil.hpp>
#include <utility>

namespace fmpi {

namespace detail {

using namespace std::literals::string_view_literals;
constexpr auto t_initialize = "Tmain.t_initialize"sv;
constexpr auto t_dispatch   = "Tmain.t_dispatch"sv;
constexpr auto t_receive    = "Tmain.t_receive"sv;
constexpr auto t_idle       = "Tmain.t_idle"sv;
constexpr auto t_compute    = kComputationTime;

struct ScheduleArgs {
  enum class window_type
  {
    SLIDING,
    FIXED
  };
  std::size_t const winsz{};
  std::string const name;
  window_type       type = window_type::FIXED;
};

class Alltoall {
 public:
  Alltoall(CollectiveArgs args, ScheduleArgs schedule);

  template <class Schedule>
  collective_future execute(Schedule schedule);

 private:
  const void* send_offset(mpi::Rank r) const;
  void*       recv_offset(mpi::Rank r) const;

  CollectiveArgs args_;
  ScheduleArgs   sched_args_;
  MPI_Aint       recvextent_{};
  MPI_Aint       sendextent_{};
};

}  // namespace detail

template <class Schedule, class InputIt, class OutputIt, size_t NReqs = 2>
inline collective_future ring_waitsome_overlap(
    InputIt begin, OutputIt out, int blocksize, mpi::Context const& ctx) {
  constexpr auto algorithm_name = std::string_view("WaitsomeOverlap");

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto collective_args = detail::CollectiveArgs{
      &*begin,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      &*out,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      ctx};

  auto schedule_args = detail::ScheduleArgs{
      NReqs,
      std::string{Schedule::NAME} + std::string{algorithm_name} +
          std::to_string(NReqs),
      detail::ScheduleArgs::window_type::SLIDING

  };

  auto coll = detail::Alltoall{collective_args, schedule_args};
  return coll.execute(Schedule{ctx});
}

template <class Schedule, class InputIt, class OutputIt, size_t NReqs = 2>
inline collective_future ring_waitall_overlap(
    InputIt begin, OutputIt out, int blocksize, mpi::Context const& ctx) {
  constexpr auto algorithm_name = std::string_view("WaitallOverlap");

  using value_type = typename std::iterator_traits<InputIt>::value_type;

  auto collective_args = detail::CollectiveArgs{
      &*begin,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      &*out,
      static_cast<std::size_t>(blocksize),
      mpi::type_mapper<value_type>::type(),
      ctx};

  auto schedule_args = detail::ScheduleArgs{
      NReqs,
      std::string{Schedule::NAME} + std::string{algorithm_name} +
          std::to_string(NReqs),
      detail::ScheduleArgs::window_type::FIXED

  };

  auto coll = detail::Alltoall{collective_args, schedule_args};
  return coll.execute(Schedule{ctx});
}

namespace detail {

inline Alltoall::Alltoall(CollectiveArgs args, ScheduleArgs schedule)
  : args_(args)
  , sched_args_(std::move(schedule)) {
  MPI_Aint recvlb{};
  MPI_Aint sendlb{};
  FMPI_ASSERT(args_.recvtype == args_.sendtype);
  MPI_Type_get_extent(args_.recvtype, &recvlb, &recvextent_);
  MPI_Type_get_extent(args_.sendtype, &sendlb, &sendextent_);
}

inline void* Alltoall::recv_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = args_.recvcount * recvextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(args_.recvbuf, offset);
}

inline const void* Alltoall::send_offset(mpi::Rank r) const {
  FMPI_ASSERT(r >= 0);
  auto const segsz  = args_.sendcount * sendextent_;
  auto const offset = static_cast<std::size_t>(r) * segsz;
  return fmpi::detail::add(args_.sendbuf, offset);
}

template <class Schedule>
inline collective_future Alltoall::execute(Schedule schedule) {
  auto const& ctx = args_.comm;

  FMPI_DBG_STREAM("running algorithm " << sched_args_.name);

  if (ctx.size() < 3) {
    auto const me = ctx.rank();
    auto const other =
        static_cast<mpi::Rank>((ctx.size()) == 1 ? me : 1 - me);
    auto ret = MPI_Sendrecv(
        send_offset(other),
        args_.sendcount,
        args_.sendtype,
        other,
        kTagRing,
        recv_offset(other),
        args_.recvcount,
        args_.recvtype,
        other,
        kTagRing,
        ctx.mpiComm(),
        MPI_STATUS_IGNORE);
    return make_ready_future(ret);
  }

  auto trace = MultiTrace{std::string_view(sched_args_.name)};

  steady_timer t_init{trace.duration(detail::t_initialize)};

  // intermediate buffer for two pipelines
  // using thread_alloc = ThreadAllocator<std::byte>;
  // auto buf_alloc     = thread_alloc{};

  auto const reqsInFlight =
      std::min(std::size_t(schedule.phaseCount()), sched_args_.winsz);

  std::array<std::size_t, detail::n_types> nslots{};
  nslots.fill(reqsInFlight);

  auto promise = collective_promise{};
  auto future  = promise.get_future();
  auto schedule_state =
      std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));

#if 0
  schedule_state->register_signal(
      message_type::IRECV,
      [buf_alloc,
       recvextent,
       recvcount = args_.recvcount,
       recvtype  = args_.recvtype](Message& message) mutable {
        auto const nbytes = recvcount * recvextent;
        auto*      buffer = buf_alloc.allocate(nbytes);
        FMPI_ASSERT(buffer);

        // add the buffer to the message
        message.set_buffer(buffer, recvcount, recvtype);
      });
#endif

  schedule_state->register_callback(
      message_type::IRECV,
      [sptr = future.arrival_queue()](std::vector<Message> msgs) mutable {
        std::move(
            std::begin(msgs), std::end(msgs), std::back_inserter(*sptr));
      });

  auto& dispatcher = static_dispatcher_pool();
  // submit into dispatcher
  auto const hdl = dispatcher.submit(std::move(schedule_state));

  t_init.finish();

  {
    auto const winsz = static_cast<uint32_t>(sched_args_.winsz);
    FMPI_DBG("Sending messages");
    steady_timer t_dispatch{trace.duration(detail::t_dispatch)};

    auto const rounds =
        std::max(tlx::div_ceil(schedule.phaseCount(), winsz), 1u);

    for (auto&& r : range(rounds)) {
      auto const last = std::min(schedule.phaseCount(), (r + 1) * winsz);

      for (auto&& rr : range(r * winsz, last)) {
        auto const rpeer = schedule.recvRank(rr);
        auto const speer = schedule.sendRank(rr);

        FMPI_DBG(std::make_pair(rpeer, speer));

        if (rpeer != ctx.rank()) {
          // auto recv = Message{rpeer, kTagRing, ctx.mpiComm()};
          auto recv = Message{
              recv_offset(rpeer),
              args_.recvcount,
              args_.recvtype,
              rpeer,
              kTagRing,
              ctx.mpiComm()};
          dispatcher.schedule(hdl, message_type::IRECV, recv);
        }

        if (speer != ctx.rank()) {
          auto send = Message{
              send_offset(speer),
              args_.sendcount,
              args_.sendtype,
              speer,
              kTagRing,
              ctx.mpiComm()};

          dispatcher.schedule(hdl, message_type::ISEND, send);
        }
      }
      if (sched_args_.type == ScheduleArgs::window_type::FIXED) {
        dispatcher.schedule(hdl, message_type::BARRIER);
      }
    }

    auto*       dst = recv_offset(ctx.rank());
    auto const* src = send_offset(ctx.rank());

    std::memcpy(dst, src, args_.sendcount * sendextent_);

    future.arrival_queue()->emplace_back(Message{
        dst,
        args_.recvcount,
        args_.recvtype,
        ctx.rank(),
        kTagRing,
        ctx.mpiComm()});
    dispatcher.commit(hdl);
  }

  return future;
}

}  // namespace detail
}  // namespace fmpi
#endif
