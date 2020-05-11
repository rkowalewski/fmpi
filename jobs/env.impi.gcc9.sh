#!/bin/sh

#Important
module load slurm_setup
module load hwloc

module unload gcc
module load gcc/9
module unload mpi.intel
module load mpi.intel/2019_gcc
