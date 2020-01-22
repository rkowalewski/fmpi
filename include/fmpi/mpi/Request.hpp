#ifndef FMPI_MPI_REQUEST_HPP
#define FMPI_MPI_REQUEST_HPP

#include <mpi.h>

#include <array>
#include <fmpi/Debug.hpp>
#include <iterator>
#include <rtlx/Assert.hpp>
#include <vector>

namespace mpi {

typedef int (*reqsome_op)(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last);

int testsome(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last);

int waitsome(
    MPI_Request* begin,
    MPI_Request* end,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last);

int waitall(
    MPI_Request* begin,
    MPI_Request* end,
    MPI_Status*  statuses = MPI_STATUSES_IGNORE);

}  // namespace mpi

#endif
