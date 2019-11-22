/*****************************************************************************
 * This file is part of the Project MPISynchronizedBarrier
 *
 * Copyright (c) 2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "MPISynchronizedBarrier.h"

#include <mpi.h>

SynchronizedBarrier::SynchronizedBarrier()
  : local_success_(0U)
{
}

SynchronizedBarrier::SynchronizedBarrier(bool local_success)
  : local_success_(static_cast<unsigned long>(local_success))
{
}

auto SynchronizedBarrier::Success(MPI_Comm comm) -> bool {

    unsigned long global_success = 0;
    MPI_Allreduce(&local_success_, &global_success, 1, MPI_UNSIGNED_LONG, MPI_LAND, comm);
    return global_success != 0U;
}

SynchronizedClock::SynchronizedClock(int sync_tag,
                                     double max_async_time,
                                     double time_to_sync)
    : sync_tag_(sync_tag)
    , max_async_time_(max_async_time)
    , time_to_sync_(time_to_sync)
    , synced_(false)
{}

auto SynchronizedClock::Init(MPI_Comm comm) -> bool {

    int nprocs;

        int myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    int synced_pes = 0;

    if (myrank == 0) {
      for (int target = 1; target != nprocs; ++target) {
        // Sync PE 'target'.
        double start_time = 0;
        double end_time   = 0;

        for (int it = 0; it != 20; ++it) {
          start_time = MPI_Wtime();
          MPI_Send(&start_time, 1, MPI_DOUBLE, target, sync_tag_, comm);
          double dummy = 0;
          MPI_Recv(
              &dummy,
              1,
              MPI_DOUBLE,
              target,
              sync_tag_,
              comm,
              MPI_STATUS_IGNORE);
          end_time = MPI_Wtime();

          if (end_time - start_time < max_async_time_) {
            synced_pes += 1;
            break;
          }
        }

        double succ = -1;
        MPI_Send(&succ, 1, MPI_DOUBLE, target, sync_tag_, comm);
      }

      time_diff_ = 0;
    }
    else {
      double time_diff = 0;
      double time      = 0;
      MPI_Recv(&time, 1, MPI_DOUBLE, 0, sync_tag_, comm, MPI_STATUS_IGNORE);
      time_diff = time - MPI_Wtime();

      while (time != static_cast<double>(-1)) {
        MPI_Send(&time, 1, MPI_DOUBLE, 0, sync_tag_, comm);
        MPI_Recv(&time, 1, MPI_DOUBLE, 0, sync_tag_, comm, MPI_STATUS_IGNORE);

        if (time != static_cast<double>(-1)) {
          time_diff = time - MPI_Wtime();
        }
      }
      time_diff_ = time_diff;
    }

    MPI_Bcast( &synced_pes, 1, MPI_INT, 0, comm );

    // std::cout << "PE " << myrank << ": " << " clock diff... " << time_diff_ << std::endl;
    synced_ = synced_pes == nprocs - 1;
    return synced_;
}

auto SynchronizedClock::Barrier(MPI_Comm comm) -> SynchronizedBarrier {

    MPI_Barrier(comm);

    double start_time = MPI_Wtime() + time_to_sync_;

    MPI_Bcast( &start_time, 1, MPI_DOUBLE, 0, comm );

    bool sync_valid = false;
    while (time_diff_ + MPI_Wtime() < start_time) {
        sync_valid = true;
    }

    return {sync_valid && synced_};
}
