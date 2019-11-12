#!/bin/bash

source "$HOME/scripts/bash-commands.sh"

procs=(8 16 32 48)

nthreads=(12 6 3 2)

for s in $(seq 0 8)
do
  for i in "${procs[@]}"
  do
    gencmdfile jobs/ng.a2a.impi.tpl -n $((2**s)) -p $i -t 1 "$@"
  done
done
