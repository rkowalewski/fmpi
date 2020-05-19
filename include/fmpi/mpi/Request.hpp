#ifndef FMPI_MPI_REQUEST_HPP
#define FMPI_MPI_REQUEST_HPP

#include <mpi.h>

#include <type_traits>

namespace mpi {

using reqsome_op = typename std::add_pointer_t<int(
    MPI_Request* first,
    MPI_Request* last,
    int*         indices,
    MPI_Status*  statuses,
    int*&        last_index)>;

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
