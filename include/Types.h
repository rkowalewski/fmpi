#ifndef TYPES_H
#define TYPES_H
#include <mpi.h>

#include <type_traits>

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

}  // namespace mpi
#endif
