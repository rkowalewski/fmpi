#!/bin/bash


# Number of procs
nprocs="$1"

# Number of cores
ncores="${2:-$(getconf _NPROCESSORS_ONLN)}"
ncpus="$((ncores / 2))"

hyperthreading="${3:-0}"

rank="$4"

# comp_cpus="$(((ncores / nprocs) - 1 - hyperthreading))"

domain_size="$((ncpus / nprocs))"
comm_cpus="1"
comp_cpus="$((domain_size - comm_cpus))"

# echo "$ncpus, $nprocs, $comp_cpus, $domain_size"

_places=""
  first_place="$((rank*domain_size + comm_cpus))"
  last_place="$((first_place + comp_cpus -1))"
  for p in $(seq $first_place $last_place)
  do
    ht="$((p+$ncpus))"
    if [[ "$hyperthreading" -eq 0 ]]; then
      _places="$_places,{$p,$ht}"
    else
      _places="$_places,{$p},{$ht}"
    fi
  done

echo "${_places#?}"




