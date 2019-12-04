#ifndef FMPI_MPI_REQUEST_H
#define FMPI_MPI_REQUEST_H

#include <mpi.h>

#include <array>
#include <iterator>
#include <vector>

#include <rtlx/Assert.hpp>

#include <fmpi/Debug.hpp>

namespace mpi {

auto waitsome(MPI_Request* begin, MPI_Request* end, int* indices) -> int*;
auto testsome(MPI_Request* begin, MPI_Request* end, int* indices) -> int*;

auto waitall(MPI_Request* begin, MPI_Request* end) -> bool;
}  // namespace mpi

#endif
