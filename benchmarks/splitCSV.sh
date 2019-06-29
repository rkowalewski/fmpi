#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

for f in $(ls -1 "${SCRIPTPATH}/results"/*.csv)
do
  myf=$f
  if [ -f "$myf" ]
  then
    head="$(head -n 1 $myf)"
    while read b
    do
      while read procs
      do
        res="$(basename $myf)"
        dir="$SCRIPTPATH/results/partials/${res%.csv}"
        mkdir -p "$dir"
        res="${dir#$(dirname $dir)/}"
        filename="$dir/${res%.csv}.${procs}.$b.csv"
        echo $head > $filename
        grep "^.*$b,[A-Za-z]\+.*${procs}\$" $myf >> $filename
      done < <(tail -n+2 $myf | awk -F, '{print $NF}' | sort -n | uniq)
    done < <(tail -n+2 $myf | awk -F, '{print $5}' | sort -n | uniq)
  fi
done

