#ifndef MPI__REQUEST_H
#define MPI__REQUEST_H

#include <mpi.h>

#include <array>
#include <iterator>
#include <vector>

#include <rtlx/Assert.h>

#include <fmpi/Debug.h>

namespace mpi {

int* waitsome(MPI_Request* begin, MPI_Request* end, int* indices);

bool waitall(MPI_Request* begin, MPI_Request* end);
}  // namespace mpi

#endif
