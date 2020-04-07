#ifndef FMPI_MPI_TYPEMAPPER_HPP
#define FMPI_MPI_TYPEMAPPER_HPP

#include <mpi.h>

#include <type_traits>

namespace mpi {

namespace detail {

template <class T>
struct type_mapper {
  static_assert(
      !std::is_trivially_copyable<T>::value,
      "MPI always requires trivially copyable types");

  static_assert(
      !std::is_arithmetic<T>::value,
      "arithmetic types can be perfectly matched to MPI Types");

  static constexpr auto type() -> MPI_Datatype {
    return MPI_BYTE;
  }

  static constexpr auto factor() -> std::size_t {
    return sizeof(T);
  }
};

#define FMPI_MPI_DATATYPE_MAPPER(integral_type, mpi_type) \
  template <>                                             \
  struct type_mapper<integral_type> {                     \
    static constexpr MPI_Datatype type() {                \
      return mpi_type;                                    \
    }                                                     \
    static constexpr std::size_t factor() {               \
      return 1;                                           \
    }                                                     \
  };

FMPI_MPI_DATATYPE_MAPPER(int, MPI_INT)
FMPI_MPI_DATATYPE_MAPPER(unsigned int, MPI_UNSIGNED)
FMPI_MPI_DATATYPE_MAPPER(char, MPI_CHAR)
FMPI_MPI_DATATYPE_MAPPER(unsigned char, MPI_UNSIGNED_CHAR)
FMPI_MPI_DATATYPE_MAPPER(short, MPI_SHORT)
FMPI_MPI_DATATYPE_MAPPER(unsigned short, MPI_UNSIGNED_SHORT)
FMPI_MPI_DATATYPE_MAPPER(long, MPI_LONG)
FMPI_MPI_DATATYPE_MAPPER(unsigned long, MPI_UNSIGNED_LONG)
FMPI_MPI_DATATYPE_MAPPER(long long, MPI_LONG_LONG)
FMPI_MPI_DATATYPE_MAPPER(unsigned long long, MPI_UNSIGNED_LONG_LONG)
FMPI_MPI_DATATYPE_MAPPER(float, MPI_FLOAT)
FMPI_MPI_DATATYPE_MAPPER(double, MPI_DOUBLE)
FMPI_MPI_DATATYPE_MAPPER(long double, MPI_LONG_DOUBLE)
FMPI_MPI_DATATYPE_MAPPER(bool, MPI_C_BOOL)

}  // namespace detail

template <class T>
struct type_mapper {
  static constexpr auto type() -> MPI_Datatype {
    return detail::type_mapper<T>::type();
  }
};

using return_code = int;
using Comm        = MPI_Comm;
using Tag         = int;

}  // namespace mpi
#endif
