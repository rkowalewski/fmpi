#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <fmpi/concurrency/Future.hpp>
#include <fmpi/mpi/Environment.hpp>

// Other AllToAll Algorithms

namespace fmpi {
/// Forward Declarations

template <class Schedule, class InputIt, class OutputIt, size_t NReqs>
void ring_waitsome(
    InputIt begin, OutputIt out, int blocksize, mpi::Context const& ctx);

template <class Schedule, class InputIt, class OutputIt, size_t NReqs>
inline void ring_waitall(
    InputIt begin, OutputIt out, int blocksize, mpi::Context const& ctx);

template <class InputIt, class OutputIt>
collective_future mpi_alltoall(
    InputIt begin, OutputIt out, int blocksize, mpi::Context const& ctx);

}  // namespace fmpi

#include <fmpi/alltoall/MpiAlltoall.hpp>
//#include <fmpi/alltoall/Waitall.hpp>
//#include <fmpi/alltoall/Waitsome.hpp>
#include <fmpi/alltoall/WaitsomeOverlap.hpp>

#endif
