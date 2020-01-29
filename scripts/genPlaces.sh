#!/bin/bash


num_procs="$1"

np="$(getconf _NPROCESSORS_ONLN)"
procs_avail="${2:-$((np/2))}"

comp_threads="$(((procs_avail / num_procs) - 1))"
comm_threads="1"

domain_size="$((comp_threads + comm_threads))"

_places=""
for d in $(seq 0 $((num_procs-1)))
do
  first_place="$((d*domain_size + comm_threads))"
  for p in $(seq $first_place $((first_place + comp_threads-1)))
  do
    _places="$_places,{$p,$((p+$procs_avail))}"
  done
done

echo "${_places#?}"



