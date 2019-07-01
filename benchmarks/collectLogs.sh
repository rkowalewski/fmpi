#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

for dir in $(ls -1 "${SCRIPTPATH}/logs")
do
  mydir="$SCRIPTPATH/logs/$dir"
  if [ -d "$mydir" ]
  then
    echo "-- analyzing directory: $mydir"
    tmpFile="$(mktemp)"
    firstFile="$(ls -l $mydir/*.out | grep '^-' | awk '{print $9}' | head -n 1)"
    # print header
    grep '^Node' "$firstFile" > "$tmpFile"
    # print all results
    grep '^[0-9]\+,' --no-filename ${mydir}/*.out >> "$tmpFile"

    # extract the directory name of the logs
    # path/to/foo/ -> foo
    res="${mydir#$(dirname $mydir)/}"

    echo "-- writing stats to results/$res.csv"

    ## collect stats
    cat "$tmpFile" | Rscript "${SCRIPTPATH}/R/stats.R" > "${SCRIPTPATH}/results/$res.csv"

    # remove temporary file to clean up everything
    rm -f "$tmpFile"
  fi
done

echo "-- splitting CSV results"
bash "$SCRIPTPATH/splitCSV.sh"

