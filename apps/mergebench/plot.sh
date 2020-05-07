#!/bin/bash

_csv_filter() {
  local _pat="$(echo $@ | sed 's#\\$#\$#g')"
  awk -F',' "(NR == 1 || ${_pat}) {print}"
}

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

tmpf=$(mktemp --tmpdir=$TMPDIR XXXXXX --suffix=".csv") ||
  {
    echo "Failed to create temp file"
    exit 1
  }

awk -f "$SCRIPTPATH/parse_csv.awk" $1 >"$tmpf"

nodes=(
  $(awk -F',' '(NR > 1) {print $4}' "$tmpf" |
    sort -nu |
    tr '\n' ' ')
)

win=(
  $(awk -F',' '(NR > 1) {print $5}' "$tmpf" |
    sort -nu |
    tr '\n' ' ')
)

_name="$(basename $1)"

git_root="$(git rev-parse --show-toplevel)"
_dirname="$(dirname $1 | sed 's#.*logs/##g')"
_target="$git_root/plots/${_dirname}"

mkdir -p "$_target"

# printf '%s\n' "${nodes[@]}"

for nn in ${nodes[@]}; do
  cat "$tmpf" |
    _csv_filter "\$4 == $nn" |
    Rscript "$SCRIPTPATH/lineplot.R" --output "$_target/$_name.$nn.pdf"
done

rm -f "$tmpf"
