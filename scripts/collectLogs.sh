#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

RES_ROOT=${SCRIPTPATH}/results

gatherStatistics() {
    mydir="$1"
    echo "-- analyzing directory: $mydir"
    tmpFile="$(mktemp)"
    firstFile="$(ls -1 $mydir/*.out | head -n 1)"
    # print header
    grep '^Node' "$firstFile" > "$tmpFile"

    for f in "$mydir"/*.out
    do
      # extract all measurements from logs
      _basename="$(basename $f)"
      _basename="${_basename%.out}"
      grep '^\([0-9]\+,\|Nodes,\)' --no-filename $f > "${mydir}/${_basename}.csv"
    done

    # print all results
    grep '^[0-9]\+,' --no-filename ${mydir}/*.out >> "$tmpFile"

    # extract the directory name of the logs
    # path/to/foo/ -> foo
    res="${mydir#$(dirname $mydir)/}"

    echo "-- writing stats to $RES_ROOT/$res.csv"

    ## collect stats
    cat "$tmpFile" | Rscript "${SCRIPTPATH}/R/stats.R" > "$RES_ROOT/$res.csv"

    # remove temporary file to clean up everything
    rm -f "$tmpFile"
}

DIR="$1"

if [ -z "$DIR" ]
then
  for dir in $(ls -1 "${SCRIPTPATH}/logs")
  do
    mydir="$SCRIPTPATH/logs/$dir"
    if [ -d "$mydir" ]
    then
      gatherStatistics "$mydir"
    fi
  done
else
  if [ ! -d "$DIR" ]
  then
    echo "not a directory: $DIR"
    exit 1
  fi
  gatherStatistics "$DIR"
fi

echo "-- splitting CSV results"
bash "$SCRIPTPATH/splitCSV.sh"

echo "-- calculating overall statistics"
if [ -z "$DIR" ]
then
  for f in $(ls -1 "$RES_ROOT")
  do
    echo "not implemented yet"
  done
else
  _dir="$(basename $DIR)"
  Rscript "$SCRIPTPATH/R/selectTop.R" "$RES_ROOT/${_dir}.csv" "$RES_ROOT/partials/${_dir}/${_dir}"
fi

