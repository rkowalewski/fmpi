#! /bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

nthreads="$(getconf _NPROCESSORS_ONLN)"
ncores="$((nthreads / 2))"

nprocs=$1

if [ $# -eq 0 ]; then
  echo "usage: $SCRIPT <nprocs> <nthreads>? <args...>?"
  exit 1
fi

nt="${2:-$((ncores / nprocs))}"

shift 2

if [ "$nt" -eq "0" ];
then
  echo "invalid number of threads"
  exit 1
fi

if [ "$((nt * nprocs))" -gt "$ncores" ]; then
  export OMP_PLACES="threads"
else
  export OMP_PLACES="cores"
fi

export OMP_NUM_THREADS="$nt"

echo "nprocs: $nprocs, nthreads: $nt"

echo "running: mpirun -n "$nprocs" ./build.release/apps/mergebench/parallel"

mpirun -n "$nprocs" ./build.release/apps/mergebench/parallel "$@"

echo "running: mpirun -n "$nprocs" ./build.release/apps/mergebench/recursive"

# mpirun -n "$nprocs" ./build.release/apps/mergebench/recursive "$@"
