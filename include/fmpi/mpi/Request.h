#ifndef FMPI_MPI_REQUEST_H
#define FMPI_MPI_REQUEST_H

#include <mpi.h>

#include <array>
#include <iterator>
#include <vector>

#include <rtlx/Assert.h>

#include <fmpi/Debug.h>

namespace mpi {

int* waitsome(MPI_Request* begin, MPI_Request* end, int* indices);
int* testsome(MPI_Request* begin, MPI_Request* end, int* indices);

bool waitall(MPI_Request* begin, MPI_Request* end);
}  // namespace mpi

#endif
