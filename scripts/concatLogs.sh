#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")



# 1: cat logs
# 2: get relevant csv lines (header + data)
# 3: remove duplicate headers
# --> concatenated csv
# 4: calculate statistics
grep '^\(Nodes\|[0-9]\+,\)' "$@" \
  | sed '2,${/^Nodes/d;}'
