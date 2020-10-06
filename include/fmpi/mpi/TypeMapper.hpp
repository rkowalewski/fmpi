#ifndef FMPI_MPI_TYPEMAPPER_HPP
#define FMPI_MPI_TYPEMAPPER_HPP

#include <mpi.h>

#include <limits>
#include <type_traits>

namespace mpi {

using return_code = int;
using Comm        = MPI_Comm;
using Tag         = int;

constexpr auto max_int =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

namespace detail {

template <class T>
struct type_mapper {
  static_assert(
      std::is_trivially_copyable<T>::value,
      "MPI always requires trivially copyable types");

  static_assert(
      std::is_arithmetic<T>::value,
      "arithmetic types can be perfectly matched to MPI Types");

  static_assert(
      !std::is_reference<T>::value,
      "We cannot map a reference type to a MPI type");

  static_assert(
      !std::is_const<T>::value, "We cannot map a const type to a MPI type");

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
    static MPI_Datatype type() {                          \
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

#undef FMPI_MPI_DATATYPE_MAPPER

}  // namespace detail

template <class T>
struct type_mapper {
  static constexpr auto type() -> MPI_Datatype {
    return detail::type_mapper<T>::type();
  }

  static constexpr auto factor() -> MPI_Datatype {
    return detail::type_mapper<T>::factor();
  }

#if 0
  static constexpr std::size_t extent() {
    MPI_Aint lb, ext;
    MPI_Type_get_extent(recvtype, &lb, &ext);
    auto ret = return sizeof(T);
  }
#endif
};

}  // namespace mpi
#endif
