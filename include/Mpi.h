#ifndef MPI_H
#define MPI_H

#include <mpi.h>

#include <array>
#include <tuple>
#include <type_traits>

#include <Debug.h>

#include <tlx/simple_vector.hpp>

namespace mpi {

using rank_t          = int;
using size_type       = MPI_Aint;
using difference_type = int;

namespace detail {
template <class T>
struct mpi_datatype {
  static_assert(
      !std::is_arithmetic<T>::value,
      "arithmetic types can be perfectly matched to MPI Types");

  static MPI_Datatype type()
  {
    return MPI_BYTE;
  }
};
template <>
struct mpi_datatype<int> {
  static MPI_Datatype type()
  {
    return MPI_INT;
  }
};

template <>
struct mpi_datatype<char> {
  static MPI_Datatype type()
  {
    return MPI_CHAR;
  }
};

template <>
struct mpi_datatype<unsigned char> {
  static MPI_Datatype type()
  {
    return MPI_UNSIGNED_CHAR;
  }
};

template <>
struct mpi_datatype<short> {
  static MPI_Datatype type()
  {
    return MPI_SHORT;
  }
};

template <>
struct mpi_datatype<unsigned short> {
  static MPI_Datatype type()
  {
    return MPI_UNSIGNED_SHORT;
  }
};

template <>
struct mpi_datatype<unsigned int> {
  static MPI_Datatype type()
  {
    return MPI_UNSIGNED;
  }
};

template <>
struct mpi_datatype<long> {
  static MPI_Datatype type()
  {
    return MPI_LONG;
  }
};

template <>
struct mpi_datatype<unsigned long> {
  static MPI_Datatype type()
  {
    return MPI_UNSIGNED_LONG;
  }
};

template <>
struct mpi_datatype<long long> {
  static MPI_Datatype type()
  {
    return MPI_LONG_LONG;
  }
};

template <>
struct mpi_datatype<unsigned long long> {
  static MPI_Datatype type()
  {
    return MPI_UNSIGNED_LONG_LONG;
  }
};

template <>
struct mpi_datatype<float> {
  static MPI_Datatype type()
  {
    return MPI_FLOAT;
  }
};

template <>
struct mpi_datatype<double> {
  static MPI_Datatype type()
  {
    return MPI_DOUBLE;
  }
};

template <>
struct mpi_datatype<long double> {
  static MPI_Datatype type()
  {
    return MPI_LONG_DOUBLE;
  }
};

}  // namespace detail

template <class T>
struct mpi_datatype {
  static MPI_Datatype type()
  {
    return detail::mpi_datatype<T>::type();
  }
};

class MpiCommCtx {
 public:
  MpiCommCtx() = default;

  MpiCommCtx(MPI_Comm const& _worldComm)
  {
    int initialized;
    A2A_ASSERT_RETURNS(MPI_Initialized(&initialized), MPI_SUCCESS);
    A2A_ASSERT(initialized == 1);

    A2A_ASSERT_RETURNS(MPI_Comm_dup(_worldComm, &m_comm), MPI_SUCCESS);
    // A2A_ASSERT_RETURNS(MPI_Comm_group(comm, &world_group), MPI_SUCCESS);

    A2A_ASSERT_RETURNS(MPI_Comm_size(m_comm, &m_size), MPI_SUCCESS);
    A2A_ASSERT_RETURNS(MPI_Comm_rank(m_comm, &m_rank), MPI_SUCCESS);

#if 0
    // split world into shared memory communicator
    A2A_ASSERT_RETURNS(
        MPI_Comm_split_type(
            comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm),
        MPI_SUCCESS);

    // setup shared memory infos
    // A2A_ASSERT_RETURNS(
    //    MPI_Comm_group(sharedComm, &shared_group), MPI_SUCCESS);
    A2A_ASSERT_RETURNS(MPI_Comm_size(sharedComm, &nrShared), MPI_SUCCESS);
    A2A_ASSERT_RETURNS(MPI_Comm_rank(sharedComm, &meShared), MPI_SUCCESS);
#endif
  }

  constexpr auto rank() const noexcept
  {
    return m_rank;
  }

  constexpr auto size() const noexcept
  {
    return m_size;
  }

  constexpr auto const& mpiComm() const noexcept
  {
    return m_comm;
  }

  MpiCommCtx(MpiCommCtx&&) noexcept = default;
  MpiCommCtx& operator=(MpiCommCtx&&) noexcept = default;

  ~MpiCommCtx()
  {
    A2A_ASSERT_RETURNS(MPI_Comm_free(&m_comm), MPI_SUCCESS);
    // A2A_ASSERT_RETURNS(MPI_Comm_free(&m_sharedComm), MPI_SUCCESS);
  }

 private:
  MPI_Comm    m_comm{MPI_COMM_NULL};
  mpi::rank_t m_size{};
  mpi::rank_t m_rank{};
};

template <class T>
inline auto mpiAllReduceMinMax(MpiCommCtx const& ctx, T value)
{
  auto mpi_type = mpi::mpi_datatype<T>::type();

  T min, max;

  A2A_ASSERT_RETURNS(
      MPI_Allreduce(&value, &min, 1, mpi_type, MPI_MIN, ctx.mpiComm()),
      MPI_SUCCESS);
  A2A_ASSERT_RETURNS(
      MPI_Allreduce(&value, &max, 1, mpi_type, MPI_MAX, ctx.mpiComm()),
      MPI_SUCCESS);

  return std::make_pair(min, max);
}

namespace detail {
struct MemorySegmentBase {
  size_type         m_nbytes{};
  difference_type   m_disp_unit{};
  MPI_Win           m_win{MPI_WIN_NULL};
  MpiCommCtx const* m_ctx;

  MemorySegmentBase() = default;

  MemorySegmentBase(MpiCommCtx const& ctx, size_t nbytes, size_t disp_unit)
    : m_nbytes(nbytes)
    , m_disp_unit(disp_unit)
    , m_ctx(&ctx)
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
    std::swap(m_ctx, other.m_ctx);
    return *this;
  }
};
}  // namespace detail

template <class T>
class ShmSegment : private detail::MemorySegmentBase {
  using detail::MemorySegmentBase::MemorySegmentBase;

  template <class _T>
  using simple_vector =
      tlx::SimpleVector<_T, tlx::SimpleVectorMode::NoInitNoDestroy>;

 public:
  using pointer       = T*;
  using const_pointer = T const*;

  ShmSegment() = default;

  ShmSegment(MpiCommCtx const& ctx, size_t nels)
    : MemorySegmentBase(ctx, nels * sizeof(T), sizeof(T))
    , m_baseptrs(m_ctx->size())
  {
    MPI_Info info = MPI_INFO_NULL;

    MPI_Info_create(&info);
    MPI_Info_set(info, "same_disp_unit", "true");

    size_t min, max;
    std::tie(min, max) = mpiAllReduceMinMax(m_ctx->mpiComm(), m_nbytes);
    if (min == max) {
      MPI_Info_set(info, "same_size", "true");
    }

    MPI_Info_set(info, "alloc_shared_noncontig", "true");

    A2A_ASSERT_RETURNS(
        MPI_Win_allocate_shared(
            m_nbytes,
            m_disp_unit,
            info,
            m_ctx->mpiComm(),
            &m_baseptrs[m_ctx->rank()],
            &m_win),
        MPI_SUCCESS);

    auto me = m_ctx->rank();

    auto queryShmPtrs = [this](auto idx) {
      size_type       sz;
      difference_type disp;

      A2A_ASSERT_RETURNS(
          MPI_Win_shared_query(m_win, idx, &sz, &disp, &m_baseptrs[idx]),
          MPI_SUCCESS);
      A2A_ASSERT(disp == m_disp_unit);
      A2A_ASSERT(m_nbytes == sz);
    };

    for (mpi::rank_t r = 0; r < me; ++r) {
      queryShmPtrs(r);
    }

    for (mpi::rank_t r = me + 1; r < m_ctx->size(); ++r) {
      queryShmPtrs(r);
    }

    A2A_ASSERT_RETURNS(MPI_Info_free(&info), MPI_SUCCESS);
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

  pointer base() noexcept
  {
    return m_baseptrs[m_ctx->rank()];
  }

  const_pointer base() const noexcept
  {
    return m_baseptrs[m_ctx->rank()];
  }

  pointer base(rank_t rank) noexcept
  {
    return m_baseptrs[rank];
  }

  const_pointer base(rank_t rank) const noexcept
  {
    return m_baseptrs[rank];
  }

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
    std::tie(min, max) = mpiAllReduceMinMax(m_ctx->mpiComm(), m_nbytes);
    if (min == max) {
      MPI_Info_set(info, "same_size", "true");
    }

    // world window
    A2A_ASSERT_RETURNS(
        MPI_Win_allocate(
            m_nbytes,
            m_disp_unit,
            MPI_INFO_NULL,
            m_ctx->mpiComm(),
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

constexpr size_t REQ_SEND = 0;
constexpr size_t REQ_RECV = 1;

template <class S, class R>
inline auto sendrecv(
    const S* sbuf,
    size_t   scount,
    int      sto,
    int      stag,
    R*       rbuf,
    size_t   rcount,
    int      rfrom,
    int      rtag,
    MPI_Comm comm)
{
  auto mpi_datatype = mpi::mpi_datatype<S>::type();

  A2A_ASSERT_RETURNS(
      MPI_Sendrecv(
          sbuf,
          static_cast<int>(scount),
          mpi_datatype,
          sto,
          stag,
          rbuf,
          static_cast<int>(rcount),
          mpi_datatype,
          rfrom,
          rtag,
          comm,
          MPI_STATUSES_IGNORE),
      MPI_SUCCESS);
}

}  // namespace mpi

#endif
