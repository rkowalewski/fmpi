#!/bin/bash

getcontext() {
  # extract version from version.h
  local gitv="$(grep FMPI_GIT_COMMIT ${2}/benchmark/src/Version.h | \
    sed 's/.*\(FMPI_GIT_COMMIT\).*"\(.*\)";/\2/g')"

  local current_date="$(date +%Y-%m-%d)"

  echo "${1}.${gitv}.${current_date}"
}

function ceiling() {
  local float_in=$1
  local ceil_val=${float_in/.*}
  local ceil_val=$((ceil_val+1))
  echo "$ceil_val"
}


getmaxblocksizel3() {
  local nprocs="$1"
  local l2cache="$(getconf LEVEL3_CACHE_SIZE)"
  local cap="$((l2cache / (nprocs)))"
  local logresult=$( echo "l($cap)/l(2)" | bc -l )
  local ceil="$(ceiling $logresult)"
  echo "$ceil"
}

getmaxblocksizel2() {
  local nprocs="$1"
  local l2cache="$(getconf LEVEL2_CACHE_SIZE)"
  local cap="$((l2cache / (nprocs)))"
  local logresult=$( echo "l($cap)/l(2)" | bc -l )
  local ceil="$(ceiling $logresult)"
  echo "$ceil"
}

getminblocksize() {
  local base="$1"
  local baseppn="$2"
  echo "$((baseppn * base / $3))"
}


source "$HOME/scripts/bash-commands.sh"

ctx=""
scale="0"
submit="0"
ppn="48"
minblocksize="4"

POSITIONAL=()

while [[ $# -gt 0 ]]
do
  key="$1"

  case $key in
      -d|--desc)
      ctx="$2"
      shift # past argument
      shift # past value
      ;;
      -s|--scale)
      scale="$2"
      shift # past argument
      shift # past value
      ;;
      -p|--procs)
      ppn="$2"
      shift # past argument
      shift # past value
      ;;
      -x|--submit)
      submit="1"
      shift # past argument
      ;;
      -l|--lower)
      minblocksize="$2"
      shift # past argument
      ;;
      *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo "positional arguments:"

for ((i = 0; i < ${#POSITIONAL[@]}; i+=2)); do
    # bash arrays are 0-indexed
    echo "   ${POSITIONAL[$i]} ${POSITIONAL[$i+1]}"
done

if [[ -z "$ctx" ]]; then
  echo "context is missing"
  exit 1
fi


git_root="$(git rev-parse --show-toplevel)"
ctx=$(getcontext "$ctx" "$git_root")

echo "$ctx"

for s in $(seq 0 $scale)
do
  nodes="$((2**s))"
  #nprocs="$((nodes * ppn))"
  #l2boundary="$(getmaxblocksizel2 $nprocs)"
  #l3boundary="$(getmaxblocksizel3 $nprocs)"
  #l3boundary="$((l3boundary+1))"
  #minblock="$(getminblocksize "$minblocksize" 48 "$ppn")"

  #echo "$l2boundary, $l3boundary, $minblock"
  # -u "$((2**l3boundary))" -l "$minblock"

  f=$(gencmdfile jobs/ng.a2a.impi.tpl \
    -n "$nodes" -p "$ppn" -d "$ctx"  -j fmpi -D "$git_root"  "${POSITIONAL[@]}")

  if [[ "$submit" == "1" ]]; then
    sbatch "$f"
  else
    echo "$f"
  fi
done
