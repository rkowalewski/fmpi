#!/bin/bash

source "$HOME/scripts/bash-commands.sh"

ctx=""

if [[ $# -gt 1 ]]; then
  ctx="$1"
  shift
fi

if [[ -z "$ctx" ]]; then
  ctx="$(date +%Y-%m-%d_%H%M%S)"
else
  ctx="${ctx}.$(date +%Y-%m-%d_%H%M%S)"
fi

procs=(16 48)

for s in $(seq 0 6)
do
  for i in "${procs[@]}"
  do
    nthreads=$((48 / i))
    gencmdfile jobs/ng.a2a.impi.tpl \
      -n $((2**s)) -p $i -t "$nthreads" -j fmpi -d "$ctx" -c "general" \
      "$@"
  done
done
