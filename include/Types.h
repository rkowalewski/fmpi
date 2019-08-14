#ifndef TYPES_H
#define TYPES_H
#include <mpi.h>

#include <type_traits>
#include <vector>

#include <Debug.h>

#include <tlx/simple_vector.hpp>

namespace mpi {
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
}  // namespace detail

template <class T>
struct mpi_datatype {
  static MPI_Datatype type()
  {
    return detail::mpi_datatype<T>::type();
  }
};

using rank_t    = int;
using size_type = MPI_Aint;

using pointer_type = MPI_Aint;

}  // namespace mpi

namespace a2a {

template <class T>
using simple_vector =
    tlx::SimpleVector<T, tlx::SimpleVectorMode::NoInitNoDestroy>;

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

 private:
  void _allocate()
  {
  }
};

template <class T>
struct MpiMemory {
  using pointer = T*;

  mpi::size_type nbytes{};
  mpi::size_type disp_unit{};

  pointer baseptr{};

  MPI_Win win{MPI_WIN_NULL};
  MPI_Win sharedWin{MPI_WIN_NULL};
  /// base pointers of all partners
  simple_vector<mpi::pointer_type> basePtrsShared;

  MpiMemory() = default;

  MpiMemory(MpiCommCtx const& ctx, size_t nels)
    : nbytes(nels * sizeof(T))
    , disp_unit(sizeof(T))
  {
    // world window
    A2A_ASSERT_RETURNS(
        MPI_Win_allocate(
            nbytes, disp_unit, MPI_INFO_NULL, ctx.comm, &baseptr, &win),
        MPI_SUCCESS);
  }

  MpiMemory& operator=(MpiMemory const& other) = delete;
  MpiMemory(MpiMemory const& other)            = delete;

  MpiMemory& operator=(MpiMemory&& other) noexcept
  {
    if (nbytes > 0) {
      MPI_Win_free(&win);
      MPI_Win_free(&sharedWin);
    }

    nbytes         = other.nbytes;
    disp_unit      = other.disp_unit;
    baseptr        = other.baseptr;
    basePtrsShared = std::move(other.basePtrsShared);

    win       = std::move(other.win);
    sharedWin = std::move(other.sharedWin);
    return *this;
  }

  MpiMemory(MpiMemory&& other) noexcept
  {
    *this = std::move(other);
  }

  ~MpiMemory()
  {
    if (nbytes > 0) {
      MPI_Win_free(&win);
    }
    baseptr   = nullptr;
    nbytes    = 0;
    disp_unit = 0;
  }
};

}  // namespace a2a
#endif
