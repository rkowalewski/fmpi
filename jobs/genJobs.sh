#!/bin/bash

source "$HOME/scripts/bash-commands.sh"

ctx=""
scale="0"
procs="48"
threads="1"

submit="0"

while getopts ":d:s:p:t:a:x" opt; do
  case ${opt} in
    d )
      ctx=$OPTARG
      ;;
    s )
      scale=$OPTARG
      ;;
    p )
      procs=$OPTARG
      ;;
    t )
      threads=$OPTARG
      ;;
    a )
      args=$OPTARG
      ;;
    x )
      submit="1"
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))


if [[ -z "$ctx" ]]; then
  echo "usage: $0 -d <context> -s <scale> -p <procs> -t <threads> -a <args>"
  exit 1
fi

git_root="$(git rev-parse --show-toplevel)"

# extract version from version.h
gitv="$(grep FMPI_GIT_COMMIT ${git_root}/benchmark/src/Version.h | \
  sed 's/.*\(FMPI_GIT_COMMIT\).*"\(.*\)";/\2/g')"

current_date="$(date +%Y-%m-%d)"

ctx="${ctx}.${gitv}.${current_date}"

for s in $(seq 0 $scale)
do
  f=$(gencmdfile jobs/ng.a2a.impi.tpl \
    -n $((2**s)) -p "$procs" -t "$threads" -j fmpi -D "$git_root" -d "$ctx" -c "general" "$args")

  if [[ "$submit" == "1" ]]; then
    sbatch "$f"
  else
    echo "$f"
  fi
done
