#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

file="$1"

if [[ "$#" -lt 1 ]]; then
  echo "arguments: $#: $@"
  echo "usage: <outfile.csv> <input...>"
  exit 1
fi

if [[ ! -f "$file" ]]; then
  echo "$file does not exist"
  exit 1
fi

if [[ ! -x "$(command -v R)" ]]; then
  echo "R seems not to be installed on your system"
  exit 1
fi

dirname="$(dirname "$file")"
filename="${file%.*}"
extension="${file##*.}"

name="$(basename $file)"
name="${name##*.}"

grep '^\([0-9]\+\s\+.*\)\|\(cpu list.*\)$' "$file" |
  sed 's/cpu.*$/\n/g' |
  awk -v RS= "{print > (\"$filename-\" NR \".log\")}"

csv="$filename.csv"

echo "bench,size,latency" >"$csv"

for f in "$dirname"/*; do
  if [[ "$f" =~ ^$filename-1 ]]; then
    data="$(awk '/^[0-9]/{printf "%d,%.2f\n", $1, $2}' "$f")"
    echo "$data" | sed 's/^/FMPI,/g' >>"$csv"
  elif [[ "$f" =~ ^$filename-2 ]]; then
    data="$(awk '/^[0-9]/{printf "%d,%.2f\n", $1, $2}' "$f")"
    echo "$data" | sed 's/^/Baseline-MT,/g' >>"$csv"
  elif [[ "$f" =~ ^$filename-3 ]]; then
    data="$(awk '/^[0-9]/{printf "%d,%.2f\n", $1, $2}' "$f")"
    echo "$data" | sed 's/^/Baseline-ST,/g' >>"$csv"
  fi
done

cat "$csv"

Rscript osu_latency.R --input "$csv" --device tikz
