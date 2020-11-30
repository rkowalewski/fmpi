/*
 * stencil_par.h
 *
 *  Created on: Jan 4, 2012
 *      Author: htor
 */

#ifndef STENCIL_PAR_H_
#define STENCIL_PAR_H_

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fmpi/concurrency/Dispatcher.hpp>
#include <fmpi/container/FixedVector.hpp>
#include <fmpi/memory/MpiAllocator.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <rtlx/ScopedLambda.hpp>

// row-major order
#define ind(i, j) (j) * (bx + 2) + (i)

void printarr_par(
    int         iter,
    double*     array,
    int         size,
    int         px,
    int         py,
    int         rx,
    int         ry,
    int         bx,
    int         by,
    int         offx,
    int         offy,
    const char* fname,
    MPI_Comm    comm);

bool setup(
    int                 argc,
    char**              argv,
    int*                n_ptr,
    int*                energy_ptr,
    int*                niters_ptr,
    int*                px_ptr,
    int*                py_ptr,
    std::string&        str,
    mpi::Context const& ctx);

#endif /* STENCIL_PAR_H_ */
