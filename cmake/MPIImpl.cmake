include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
include(CMakePrintHelpers)

list(APPEND CMAKE_REQUIRED_INCLUDES "${MPI_CXX_INCLUDE_DIRS}")

  # check for Open MPI
check_cxx_symbol_exists(
  OMPI_MAJOR_VERSION
  mpi.h
  HAVE_OPEN_MPI
)

if (HAVE_OPEN_MPI)
    set(MPI_IMPL_ID "ompi")
endif()

# order matters: all of the following
# implementations also define MPICH
if (NOT DEFINED MPI_IMPL_ID)
  # check for Intel MPI
  check_cxx_symbol_exists(
    I_MPI_VERSION
    mpi.h
    HAVE_I_MPI
  )
endif()

if (HAVE_I_MPI)
    set(MPI_IMPL_ID "impi")
endif()

if (NOT DEFINED MPI_IMPL_ID)
  # check for MVAPICH
  check_cxx_symbol_exists(
    MVAPICH2_VERSION
    mpi.h
    HAVE_MVAPICH
  )
endif()

if (HAVE_MVAPICH)
    set(MPI_IMPL_ID "mvapich")
endif()

if (NOT DEFINED MPI_IMPL_ID)
  # check for MVAPICH
  check_cxx_symbol_exists(
    MPICH
    mpi.h
    HAVE_MPICH
  )
endif()

if (HAVE_MPICH)
    set(MPI_IMPL_ID "mpich")
endif()

message(INFO " found mpi library name: ${MPI_IMPL_ID}")

list(REMOVE_ITEM CMAKE_REQUIRED_INCLUDES "${MPI_CXX_INCLUDE_DIRS}")
