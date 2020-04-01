#ifndef FMPI_ALLTOALL_HPP
#define FMPI_ALLTOALL_HPP

#include <mpi.h>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>
#include <fmpi/NumericRange.hpp>

#include <fmpi/mpi/Algorithm.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <tlx/simple_vector.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/Trace.hpp>

// Other AllToAll Algorithms

namespace fmpi {

/// Forward Declarations

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
void scatteredPairwiseWaitsome(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
inline void scatteredPairwiseWaitsomeOverlap(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <
    class Schedule,
    class InputIt,
    class OutputIt,
    class Op,
    size_t NReqs>
inline void scatteredPairwiseWaitall(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

template <class InputIt, class OutputIt, class Op>
inline void MpiAlltoAll(
    InputIt             begin,
    OutputIt            out,
    int                 blocksize,
    mpi::Context const& ctx,
    Op&&                op);

}  // namespace fmpi

#include <fmpi/alltoall/MpiAlltoall.hpp>
#include <fmpi/alltoall/Waitall.hpp>
#include <fmpi/alltoall/Waitsome.hpp>

#endif
