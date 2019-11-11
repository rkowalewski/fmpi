#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

FILE="$1"

if [[ ! -x "$(command -v Rscript)" ]]
then
  echo "R seems not to be installed on your system"
  exit 1
fi

shift # remove file

if [[ -f "$FILE" && ! -w "$FILE" ]]
then
  echo "file not writable: $FILE"
  return 1
fi

if [[ ! -e "$FILE" ]]
then
  if [[ "${FILE##*.}" != "csv" ]]
  then
    echo "output file must be csv: $FILE"
    return 1
  elif [[ ! -w "$(dirname "$FILE")" ]]
  then
    echo "path to file $FILE not writable"
    return 1
  fi
fi


append="0"
if [[ -w "$FILE" ]]
then
  append="1"
fi

# 1: get relevant csv lines (header + data)
# 2: remove duplicate headers
# --> concatenated csv
# 3: calculate statistics
grep -h '^\(Nodes\|[0-9]\+,\)' "$@" \
  | sed '2,${/^Nodes/d;}' \
  | Rscript "$SCRIPTPATH/R/stats.R" "$FILE" "$append"
