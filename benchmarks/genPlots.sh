#!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

DIR="$1"

if [ ! -d "$DIR" ]
then
  echo "not a directory: $DIR"
  exit 1
fi

for f in $(ls -1 "$DIR/"*.csv)
do
  name="$(basename $f)"
  fullpath="$DIR/${name%.csv}.pdf"

  Rscript "$SCRIPTPATH/R/plot.R" "$f" "$fullpath"
done
