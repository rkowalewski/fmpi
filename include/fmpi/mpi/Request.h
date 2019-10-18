#ifndef MPI__REQUEST_H
#define MPI__REQUEST_H

#include <mpi.h>

namespace mpi {

inline std::vector<int> waitsome(MPI_Request* begin, MPI_Request* end);

}

#endif
