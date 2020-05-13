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

procs=(
  $(awk -F',' '(NR > 1) {print $4}' "$tmpf" |
    sort -nu |
    tr '\n' ' ')
)

threads=(
  $(awk -F',' '(NR > 1) {print $5}' "$tmpf" |
    sort -nu |
    tr '\n' ' ')
)

echo "$tmpf"

printf '%s\n' "${procs[@]}"
printf '%s\n' "${threads[@]}"

_name="$(basename $1)"

git_root="$(git rev-parse --show-toplevel)"
_dirname="$(dirname $1 | sed 's#.*logs/##g')"
_target="$git_root/plots/${_dirname}"

sizes=("4096")
nsizes="${#sizes[@]}"
sizes_label=("small")

mkdir -p "$_target"

lastsize="0"

for np in ${procs[@]}; do
  for nt in ${threads[@]}; do
    min_filter="\$6 > $lastsize"
    for i in "${!sizes[@]}"; do
      sz="${sizes[i]}"
      filter="\$4 == $np && \$5 == $nt && $min_filter && \$6 <= $sz"
      cat "$tmpf" |
        _csv_filter "$filter" |
        Rscript "$SCRIPTPATH/lineplot.R" \
          --output "$_target/$_name.$np.$nt.${sizes_label[i]}.pdf"
      lastsize="$sz"
    done
    filter="\$4 == $np && \$5 == $nt && \$6 > ${sizes[$((nsizes - 1))]}"
    cat "$tmpf" |
      _csv_filter "$filter" |
      Rscript "$SCRIPTPATH/lineplot.R" --output "$_target/$_name.$np.$nt.pdf"
    lastsize="0"
  done
done

rm -f "$tmpf"
