#!/bin/bash
#@ energy_policy_tag=NONE
#@ minimize_time_to_solution=yes

#-- Intel MPI: ------------
#@ job_type=MPICH
#--------------------------

#@ class=<<CLASS>>

#@ node=<<NUM_NODES>>
#@ tasks_per_node=<<NUM_PROCS>>

#@ node_usage=not_shared

# not needed for class general:
#@ island_count=1

#@ wall_clock_limit=<<WCLIMIT>>
#@ job_name=<<JOB_NAME>>
#@ network.MPI=sn_all,not_shared,us
#@ initialdir=/home/hpc/pr92fo/di25qoy/workspaces/alltoall
#@ output=/home/hpc/pr92fo/di25qoy/logs/$(job_name)/job.n<<NUM_NODES>>p<<NUM_PROCS>>t<<NUM_THREADS>>.$(schedd_host).$(jobid).out
#@ error=/home/hpc/pr92fo/di25qoy/logs/$(job_name)/job.n<<NUM_NODES>>p<<NUM_PROCS>>t<<NUM_THREADS>>.$(schedd_host).$(jobid).err
#@ notification=error
#@ notify_user=kowalewski@nm.ifi.lmu.de
#@ queue

#Intel MPI Specific Settings
#export I_MPI_SCALABLE_OPTIMIZATION=off
#export I_MPI_ADJUST_BARRIER=4

# Setup environment:
. /etc/profile
. /etc/profile.d/modules.sh

#-- Intel MPI: ------------
module switch mpi.ibm mpi.intel/2018

# Other Libraries
# module load hwloc/1.11
# module load papi/5.6
# module load likwid
# module load tbb/2018

# for newest libstdc++
module load gcc/7
#--------------------------

export DASH_ENABLE_TRACE=1

export OMP_NUM_THREADS=<<NUM_THREADS>>

## we can disable threading at all for units
##export DASH_DISABLE_THREADS=0

# we enable hyperthreading by default
export DASH_MAX_SMT=1

## We can specify a maximum number of threads per unit
export DASH_MAX_UNIT_THREADS=<<NUM_THREADS>>

#-genv  I_MPI_PIN "on" -genv I_MPI_PIN_DOMAIN "numa"

mpiexec \
    -n $((<<NUM_PROCS>> * <<NUM_NODES>>)) ./build/MpiAlltoAllBench "<<NUM_NODES>>"

