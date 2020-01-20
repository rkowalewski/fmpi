#!/bin/bash

node_size="48"
num_procs="$1"

comp_threads="$(((node_size / num_procs) - 1))"
comm_threads="1"

domain_size="$((comp_threads + comm_threads))"

_places=""
for d in $(seq 0 $((num_procs-1)))
do
  first_place="$((d*domain_size + comm_threads))"
  for p in $(seq $first_place $((first_place + comp_threads-1)))
  do
    _places="$_places,{$p,$((p+48))}"
  done
done

echo "${_places#?}"



