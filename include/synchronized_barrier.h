/*****************************************************************************
 * This file is part of the Project MPISynchronizedBarrier
 * 
 * Copyright (c) 2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#pragma once

#include <mpi.h>

class SynchronizedBarrier {
public:
    SynchronizedBarrier();

    bool Success(MPI_Comm comm);

protected:
    friend class SynchronizedClock;

    SynchronizedBarrier(bool local_success);

    unsigned long local_success_;

};

class SynchronizedClock {
public:
    SynchronizedClock(int sync_tag = 01101,
                      double max_async_time = 0.000005,
                      double time_to_sync = 0.1);

    bool Init(MPI_Comm comm);
    
    SynchronizedBarrier Barrier(MPI_Comm comm);

protected:
    int sync_tag_;
    double max_async_time_;
    double time_to_sync_;
    double time_diff_;
    bool synced_;
};
