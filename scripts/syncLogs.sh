#!/bin/sh


root="$(git rev-parse --show-toplevel)"

smuc_src='skx.supermuc:~/workspaces/alltoall'
proj_src='projekt03:~/workspaces/MpiAllToAllAlgos'

smuc_dst="$root/logs/skx.supermuc/"
proj_dst="$root/logs/projekt03/"

mkdir -p "$smuc_dst"
mkdir -p "$proj_dst"

rsync -avzP --cvs-exclude --progress --delete --exclude="**~" "$smuc_src/logs/" "$smuc_dst"
rsync -avzP --cvs-exclude --progress --delete --exclude="**~" "$proj_src/logs/" "$proj_dst"
