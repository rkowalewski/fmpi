#ifndef MPI_H
#define MPI_H

#include <mpi.h>

#include <array>
#include <type_traits>

#include <Debug.h>

#include <tlx/simple_vector.hpp>

namespace mpi {

using rank_t    = int;
using size_type = MPI_Aint;

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

struct MpiCommCtx {
  MPI_Comm    comm{MPI_COMM_NULL};
  mpi::rank_t nr{};
  mpi::rank_t me{};
  // MPI_Group   world_group{MPI_GROUP_NULL};

  MPI_Comm    sharedComm{MPI_COMM_NULL};
  mpi::rank_t nrShared{};
  mpi::rank_t meShared{};
  // MPI_Group   shared_group{MPI_GROUP_NULL};

  MpiCommCtx() = default;

  MpiCommCtx(MPI_Comm const& _worldComm)
  {
    int initialized;
    A2A_ASSERT_RETURNS(MPI_Initialized(&initialized), MPI_SUCCESS);
    A2A_ASSERT(initialized == 1);

    A2A_ASSERT_RETURNS(MPI_Comm_dup(_worldComm, &comm), MPI_SUCCESS);
    // A2A_ASSERT_RETURNS(MPI_Comm_group(comm, &world_group), MPI_SUCCESS);

    A2A_ASSERT_RETURNS(MPI_Comm_size(comm, &nr), MPI_SUCCESS);
    A2A_ASSERT_RETURNS(MPI_Comm_rank(comm, &me), MPI_SUCCESS);

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
  }

  ~MpiCommCtx()
  {
    A2A_ASSERT_RETURNS(MPI_Comm_free(&comm), MPI_SUCCESS);
    A2A_ASSERT_RETURNS(MPI_Comm_free(&sharedComm), MPI_SUCCESS);
  }
};

template <class T>
struct SharedMpiMemory {
  using pointer = T*;

  mpi::size_type    nbytes{};
  mpi::size_type    disp_unit{};
  pointer           baseptr{};
  MpiCommCtx const* ctx{};

  MPI_Win win{MPI_WIN_NULL};
  /// base pointers of all partners
  tlx::SimpleVector<pointer, tlx::SimpleVectorMode::NoInitNoDestroy>
      basePtrsShared{};

  SharedMpiMemory() = default;

  SharedMpiMemory(MpiCommCtx const& _ctx, size_t _nels)
    : nbytes(_nels * sizeof(T))
    , disp_unit(sizeof(T))
    , ctx(&_ctx)
    , basePtrsShared(ctx->nrShared)
  {
    MPI_Info info = MPI_INFO_NULL;

    A2A_ASSERT(basePtrsShared.size() > 0);

    MPI_Info_create(&info);

    size_t min, max;

    auto mpi_type = mpi::mpi_datatype<size_t>::type();
    MPI_Allreduce(&nbytes, &min, 1, mpi_type, MPI_MIN, ctx->comm);
    MPI_Allreduce(&nbytes, &max, 1, mpi_type, MPI_MAX, ctx->comm);

    MPI_Info_set(info, "same_disp_unit", "true");

    if (min == max) {
      MPI_Info_set(info, "same_size", "true");
    }

    MPI_Info_set(info, "alloc_shared_noncontig", "true");

    A2A_ASSERT_RETURNS(
        MPI_Win_allocate_shared(
            nbytes,
            disp_unit,
            info,
            ctx->sharedComm,
            &basePtrsShared[ctx->meShared],
            &win),
        MPI_SUCCESS);

    baseptr = basePtrsShared[ctx->meShared];

    for (mpi::rank_t r = 0; r < ctx->nrShared; ++r) {
      if (r == ctx->meShared) continue;

      mpi::size_type sz;
      int            disp;

      A2A_ASSERT_RETURNS(
          MPI_Win_shared_query(win, r, &sz, &disp, &basePtrsShared[r]),
          MPI_SUCCESS);
      A2A_ASSERT(static_cast<mpi::size_type>(disp) == disp_unit);
      A2A_ASSERT(nbytes == sz);
    }

    A2A_ASSERT_RETURNS(MPI_Info_free(&info), MPI_SUCCESS);
  }

  SharedMpiMemory& operator=(SharedMpiMemory const& other) = delete;
  SharedMpiMemory(SharedMpiMemory const& other)            = delete;

  SharedMpiMemory& operator=(SharedMpiMemory&& other) noexcept
  {
    if (win != MPI_WIN_NULL) {
      A2A_ASSERT_RETURNS(MPI_Win_free(&win), MPI_SUCCESS);
    }

    nbytes         = other.nbytes;
    disp_unit      = other.disp_unit;
    baseptr        = other.baseptr;
    basePtrsShared = std::move(other.basePtrsShared);

    win = std::move(other.win);
    return *this;
  }

  SharedMpiMemory(SharedMpiMemory&& other) noexcept
  {
    *this = std::move(other);
  }

  ~SharedMpiMemory()
  {
    if (win != MPI_WIN_NULL) {
      A2A_ASSERT_RETURNS(MPI_Win_free(&win), MPI_SUCCESS);
    }
    baseptr = nullptr;
  }
};

template <class T>
struct MpiMemory {
  using pointer = T*;

  mpi::size_type    nbytes{};
  mpi::size_type    disp_unit{};
  MpiCommCtx const* ctx;

  pointer baseptr{};

  MPI_Win win{MPI_WIN_NULL};

  MpiMemory() = default;

  MpiMemory(MpiCommCtx const& _ctx, size_t _nels)
    : nbytes(_nels * sizeof(T))
    , disp_unit(sizeof(T))
    , ctx(&_ctx)
  {
    // world window
    A2A_ASSERT_RETURNS(
        MPI_Win_allocate(
            nbytes, disp_unit, MPI_INFO_NULL, ctx->comm, &baseptr, &win),
        MPI_SUCCESS);
  }

  MpiMemory& operator=(MpiMemory const& other) = delete;
  MpiMemory(MpiMemory const& other)            = delete;

  MpiMemory& operator=(MpiMemory&& other) noexcept
  {
    if (win != MPI_WIN_NULL) {
      A2A_ASSERT_RETURNS(MPI_Win_free(&win), MPI_SUCCESS);
    }

    nbytes    = other.nbytes;
    disp_unit = other.disp_unit;
    baseptr   = other.baseptr;

    win = std::move(other.win);
    return *this;
  }

  MpiMemory(MpiMemory&& other) noexcept
  {
    *this = std::move(other);
  }

  ~MpiMemory()
  {
    if (win != MPI_WIN_NULL) {
      A2A_ASSERT_RETURNS(MPI_Win_free(&win), MPI_SUCCESS);
    }
    baseptr   = nullptr;
    nbytes    = 0;
    disp_unit = 0;
  }
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
