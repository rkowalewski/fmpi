#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

OUTFILE="$1"

if [[ "$#" -lt 2 ]]; then
  echo "arguments: $#: $@"
  echo "usage: <outfile.csv> <input...>"
  exit 1
fi

if [[ "${OUTFILE##*.}" != "csv" ]]; then
  echo "<outfile> must be csv: $OUTFILE"
  exit 1
fi

if [[ ! -x "$(command -v R)" ]]; then
  echo "R seems not to be installed on your system"
  exit 1
fi

shift # remove file

if ! mkdir -p "$(dirname "$OUTFILE")"; then
    echo "path to file $OUTFILE not writable"
    exit
fi

TMPDIR="/tmp"

if [[ -n "$SCRATCH" && -d "$SCRATCH" ]]; then
  TMPDIR="$SCRATCH/tmp"
  mkdir -p "$TMPDIR"
fi

files=($(ls -1 "$@"))

if [[ ${#files[@]} -lt 1 ]]; then
  echo "no log files to process"
  exit 0
fi

echo "list of files:"

for f in "${files[@]}"; do
  echo "${f}"
done

echo ""

TMPF=$(mktemp --tmpdir=$TMPDIR XXXXXX --suffix=".csv") ||
  {
    echo "Failed to create temp file"
    exit 1
  }

# 1: get relevant csv lines (header + data)
# 2: remove duplicate headers
# --> concatenated csv
# 3: calculate statistics

echo "-- collecting statistics in $TMPF"

grep -h '^\(Nodes\|[0-9]\+,\)' "${files[@]}" |
  sed '2,${/^Nodes/d;}' >"$TMPF"

echo "-- summarizing statistics in $OUTFILE"

Rscript "$SCRIPTPATH/R/stats.R" --input "$TMPF" --output "$OUTFILE"
#
# if [[ $? -eq 0 ]]; then
#   info_file="${OUTFILE%.csv}-meta.txt"
#   echo "- temp file: $TMPF" > "$info_file"
#   echo "- logs:" >> "$info_file"
#   printf "  %s\n" "${files[@]}" >> "$info_file"
# fi
