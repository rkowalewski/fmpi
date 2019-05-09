#ifndef TYPES_H__INCLUDED
#define TYPES_H__INCLUDED
#include <mpi.h>

#include <type_traits>

namespace mpi {
namespace detail {
template <class T>
struct mpi_datatype {
  static_assert(
      !std::is_arithmetic<T>::value,
      "arithmetic types can be perfectly matched to MPI Types");

  static constexpr const MPI_Datatype value = MPI_BYTE;
};
template <>
struct mpi_datatype<int> {
  static constexpr const MPI_Datatype value = MPI_INT;
};
}  // namespace detail
template <class T>
struct mpi_datatype {
  static constexpr const MPI_Datatype value =
      detail::mpi_datatype<std::remove_cv_t<T>>::value;
};

}  // namespace mpi
#endif
