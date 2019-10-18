#ifndef MPI_H
#define MPI_H

#include <mpi.h>

#include <array>
#include <tuple>
#include <type_traits>

#include <rtlx/Assert.h>

#include <tlx/simple_vector.hpp>

#include <fmpi/mpi/Algorithm.h>

#if 0
namespace mpi {

namespace detail {

struct MemorySegmentBase {
  size_type         m_nbytes{};
  difference_type   m_disp_unit{};
  MPI_Win           m_win{MPI_WIN_NULL};

  MemorySegmentBase() = default;

  MemorySegmentBase(size_t nbytes, size_t disp_unit)
    : m_nbytes(nbytes)
    , m_disp_unit(disp_unit)
  {
  }

  ~MemorySegmentBase()
  {
    if (m_win != MPI_WIN_NULL) {
      MPI_Win_free(&m_win);
    }
  }

  MemorySegmentBase(MemorySegmentBase const&) = delete;
  MemorySegmentBase& operator=(MemorySegmentBase const&) = delete;

  MemorySegmentBase(MemorySegmentBase&& other) noexcept
  {
    *this = std::move(other);
  }

  MemorySegmentBase& operator=(MemorySegmentBase&& other) noexcept
  {
    std::swap(m_win, other.m_win);
    std::swap(m_nbytes, other.m_nbytes);
    std::swap(m_disp_unit, other.m_disp_unit);
    return *this;
  }

 public:
  constexpr MPI_Win const& win() const noexcept
  {
    return m_win;
  }
};
}  // namespace detail

template <class T>
class ShmSegment : private detail::MemorySegmentBase {
  using base_t = detail::MemorySegmentBase;

  using base_t::MemorySegmentBase;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

 public:
  using pointer       = T*;
  using const_pointer = T const*;

  ShmSegment() = default;

  ShmSegment(MpiCommCtx const & ctx, size_t nels)
    : MemorySegmentBase(nels * sizeof(T), sizeof(T))
    , m_baseptrs(ctx.size())
  {
    MPI_Info info = MPI_INFO_NULL;

    MPI_Info_create(&info);
    MPI_Info_set(info, "same_disp_unit", "true");

    size_t min, max;
    std::tie(min, max) = allreduce_minmax(ctx.mpiComm(), m_nbytes);
    if (min == max) {
      MPI_Info_set(info, "same_size", "true");
    }

    MPI_Info_set(info, "alloc_shared_noncontig", "true");

    RTLX_ASSERT_RETURNS(
        MPI_Win_allocate_shared(
            m_nbytes,
            m_disp_unit,
            info,
            ctx.mpiComm(),
            &m_baseptrs[ctx.rank()],
            &m_win),
        MPI_SUCCESS);

    auto me = ctx.rank();

    auto queryShmPtrs = [this](auto idx) {
      size_type       sz;
      difference_type disp;

      RTLX_ASSERT_RETURNS(
          MPI_Win_shared_query(m_win, idx, &sz, &disp, &m_baseptrs[idx]),
          MPI_SUCCESS);
      RTLX_ASSERT(disp == m_disp_unit);
      RTLX_ASSERT(m_nbytes == sz);
    };

    for (mpi::rank_t r = 0; r < me; ++r) {
      queryShmPtrs(r);
    }

    for (mpi::rank_t r = me + 1; r < ctx.size(); ++r) {
      queryShmPtrs(r);
    }

    RTLX_ASSERT_RETURNS(MPI_Info_free(&info), MPI_SUCCESS);
  }

  ShmSegment& operator=(ShmSegment&& other) noexcept
  {
    MemorySegmentBase::operator=(std::move(other));

    m_baseptrs = std::move(other.m_baseptrs);

    return *this;
  }

  ShmSegment(ShmSegment&& other) noexcept
  {
    *this = std::move(other);
  }

  pointer base(rank_t rank) noexcept
  {
    return m_baseptrs[rank];
  }

  const_pointer base(rank_t rank) const noexcept
  {
    return m_baseptrs[rank];
  }

  using base_t::win;

 private:
  /// base pointers of all partners
  simple_vector<pointer> m_baseptrs{};
};

template <class T>
struct GlobalSegment : private detail::MemorySegmentBase {
  using detail::MemorySegmentBase::MemorySegmentBase;

 public:
  using pointer       = T*;
  using const_pointer = T const*;

  GlobalSegment() = default;

  GlobalSegment(MpiCommCtx const& ctx, size_t nels)
    : MemorySegmentBase(ctx, nels * sizeof(T), sizeof(T))
  {
    MPI_Info info = MPI_INFO_NULL;

    MPI_Info_create(&info);

    MPI_Info_create(&info);
    MPI_Info_set(info, "same_disp_unit", "true");

    size_t min, max;
    std::tie(min, max) = allreduce_minmax(ctx, m_nbytes);
    if (min == max) {
      MPI_Info_set(info, "same_size", "true");
    }

    // world window
    RTLX_ASSERT_RETURNS(
        MPI_Win_allocate(
            m_nbytes,
            m_disp_unit,
            MPI_INFO_NULL,
            ctx.mpiComm(),
            &m_baseptr,
            &m_win),
        MPI_SUCCESS);
  }

  GlobalSegment& operator=(GlobalSegment const& other) = delete;
  GlobalSegment(GlobalSegment const& other)            = delete;

  GlobalSegment& operator=(GlobalSegment&& other) noexcept
  {
    MemorySegmentBase::operator=(std::move(other));

    m_baseptr = std::move(other.m_baseptr);

    return *this;
  }

  GlobalSegment(GlobalSegment&& other) noexcept
  {
    *this = std::move(other);
  }

  pointer base() noexcept
  {
    return m_baseptr;
  }

  const_pointer base() const noexcept
  {
    return m_baseptr;
  }

 private:
  pointer m_baseptr;
};


}  // namespace mpi
#endif

#endif
