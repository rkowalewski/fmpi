#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

distinct_values() {
  field="$1"
  file="$2"

  awk -F, "{ if (NR > 1) { a[\$$field]++}} END { for (b in a) { print b } }" $file
}

for f in $(ls -1 "${SCRIPTPATH}/results"/*.csv)
do
  myf=$f
  if [ -f "$myf" ]
  then
    head="$(head -n 1 $myf)"
    while read blocksize
    do
      while read procs
      do
        res="$(basename $myf)"
        dir="$SCRIPTPATH/results/partials/${res%.csv}"
        mkdir -p "$dir"
        res="${dir#$(dirname $dir)/}"
        filename="$dir/${res%.csv}.${procs}.$blocksize.csv"
        echo $head > $filename
        grep "^.*$blocksize,[A-Za-z]\+.*${procs}\$" $myf >> $filename
      done < <(distinct_values "NF" "$myf")
    done < <(distinct_values "5" "$myf")
  fi
done

