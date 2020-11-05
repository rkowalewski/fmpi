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

shift

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

name="$(basename "$file")"
name="${name%.*}"

nodes="$(echo $name | sed 's#.*\.n\([0-9]\+\).*#\1#')"
procs="$(echo $name | sed 's#.*\.p\([0-9]\+\).*#\1#')"

sed 's#^\(\[0\].*version.*$\)#\n\1#g;s#^\(\-\-.*$\)#\n\1#g' "$file" |
  sed -n '/^#\sSize/,/^$/p;' |
   awk -v RS= "{print > (\"$filename-\" NR \".log\")}"

csv="$name.csv"

echo "nodes,procs,bench,size,total,compute,init,mpi.test,mpi.wait,comm,overlap," \
  "fmpi.waitall,fmpi.testall,fmpi.waitany,fmpi.dispatch,fmpi.copy" >"$csv"

pattern="${name//\./\\.}"

variants=("blocking" "nonblocking" "nonblocking")
# algos=("ring" "of" "bruck")
algos=("Ring" "OneFactor")
winsz=("1" "64" "4")

for f in "$dirname"/*"$name"*.log; do
  num="$(echo "$f" | sed 's#.*'"$pattern"'-\([0-9]\+\).*#\1#g')"

  v_idx="$(((num - 1) / 2))"
  w_idx="$v_idx"

  if [[ "$num" -lt 7 ]]; then
    a_idx="$(((num - 1) % 2))"
    data="$(awk '/^[0-9]/{printf "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12 , $13}' "$f")"
    data="$(echo "$data" | sed 's/^/'"${algos[$a_idx]}.${variants[$v_idx]}.${winsz[$w_idx]}"',/g')"
    #   elif [[ "$num" -lt 7 ]]; then
    #     data="$(awk '/^[0-9]/{printf "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12 , $13}' "$f")"
    #     echo "$data" | sed 's/^/'"${algos[$((num - 4))]}.nonblocking"',/g' >>"$csv"
  else
    data="$(awk '/^[0-9]/{printf "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", $1, $2, $3, $4, $5, $6, $7, $8}' "$f")"
    data="$(echo "$data" | sed 's/^/Baseline,/g')"
  fi

  echo "$data" | sed "s/^/$nodes,$procs,/g" >> "$csv"
done

# Rscript "$SCRIPTPATH"/osu_ialltoall.R --input "$csv" "$@"
# Rscript "$SCRIPTPATH"/osu_ialltoall.R --input "$csv" "$@" --speedup

echo "generated plots for $name"
