#!/bin/bash

csv_filter() {
  local _pat="$(echo $@ | sed 's#\\$#\$#g')"
  awk -F',' "(NR == 1 || ${_pat}) {print}"
}

input="$1"

patterns=("Overlap" "Waitsome[0-9]" "Waitall" "Bruck")
baseline="AlltoAll"

nodes=(
  $(awk -F',' '{if (NR > 1) {print $1}}' "$input" |
    sort -nu |
    tr '\n' ' ')
)

procs="$(awk -F',' '{if (NR == 2) {print $2}}' $input)"
threads="$(awk -F',' '{if (NR == 2) {print $3}}' $input)"
ppn="$(awk -F',' '{if (NR == 2) {print $NF}}' $input)"


echo "$procs $threads $ppn"

git_root="$(git rev-parse --show-toplevel)"

_dirname="$(dirname $input)"
_target="$git_root/plots/${_dirname##*/}"

mkdir -p $_target

for algo in ${patterns[@]}; do
  for nn in ${nodes[@]}; do
    _desc="$(echo $algo | sed 's/\[.*\]//')"
    _plot="$_target/${_desc}.n${nn}.pdf"
    cat "$input" | csv_filter "\$1 == $nn && \$5 ~ /${baseline}|${algo}/" |
      Rscript "$git_root/scripts/R/linePlot.R" \
        --output "$_plot" \
        --caption "Nodes: $nn, procs: $procs, ppn: $ppn, threads: $threads" \

    if [[ $? -eq 0 ]]; then
      echo "--generated plot: $_plot"
    fi
  done
done
