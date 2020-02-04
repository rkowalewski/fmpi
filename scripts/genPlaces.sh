#!/bin/bash


# Number of procs
nprocs="$1"

# Number of cores
np="$(getconf _NPROCESSORS_ONLN)"
ncores="${2:-$((np/2))}"

comp_threads="$(((ncores / nprocs) - 1))"
comm_threads="1"

domain_size="$((comp_threads + comm_threads))"

_places=""
for d in $(seq 0 $((nprocs-1)))
do
  first_place="$((d*domain_size + comm_threads))"
  for p in $(seq $first_place $((first_place + comp_threads-1)))
  do
    _places="$_places,{$p,$((p+$ncores))}"
  done
done

echo "${_places#?}"



