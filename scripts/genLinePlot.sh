#!/bin/bash

# allow pattern to expand to an empty string
# shopt -s nullglob

_csv_filter() {
  local _pat="$(echo $@ | sed 's#\\$#\$#g')"
  awk -F',' "(NR == 1 || ${_pat}) {print}"
}

_ls_pattern() {
  if ls *"$1" 1>/dev/null 2>&1; then
    ls *"$1" | xargs
  fi
}

input="$1"

if [[ ! -f $input ]]; then
  echo "input file missing"
  exit 1
fi

patterns=("Overlap" "Waitsome[0-9]" "Waitall" "Bruck")
baseline="AlltoAll"

nodes=(
  $(awk -F',' '{if (NR > 1) {print $1}}' "$input" |
    sort -nu |
    tr '\n' ' ')
)

threads="$(awk -F',' '{if (NR == 2) {print $3}}' $input)"
ppn="$(awk -F',' '{if (NR == 2) {print $NF}}' $input)"

git_root="$(git rev-parse --show-toplevel)"

_dirname="$(dirname $input)"
_target="$git_root/plots/${_dirname##*/}"
_filename="$(basename $input)"

mkdir -p $_target

for algo in ${patterns[@]}; do
  for nn in ${nodes[@]}; do
    procs="$((nn * ppn))"
    _desc="$(echo $algo | sed 's/\[.*\]//')"
    _plot="$_target/${_filename%.csv}.${_desc}.n${nn}.pdf"
    _caption="${_filename%.csv} (#N: $nn, #P: $procs, #PPN: $ppn, #T: $threads)"

    cat "$input" | _csv_filter "\$1 == $nn && \$5 ~ /${baseline}|${algo}/" |
      Rscript "$git_root/scripts/R/linePlot.R" \
        --output "$_plot" \
        --caption "$_caption"

    if [[ $? -eq 0 ]]; then
      echo "--generated plot: $_plot"
    fi
  done
done

cd "$_target"

for nn in ${nodes[@]}; do
  pdfjam \
    $(_ls_pattern "Waitsome.n${nn}.pdf") \
    $(_ls_pattern "Overlap.n${nn}.pdf") \
    $(_ls_pattern "Waitall.n${nn}.pdf") \
    $(_ls_pattern "Bruck.n${nn}.pdf") \
    --landscape --frame true --nup 3x2 --outfile "all.n${nn}.pdf"
done
