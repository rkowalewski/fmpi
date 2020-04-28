#!/bin/sh


root="$(git rev-parse --show-toplevel)"
remote="skx.supermuc:~/workspaces/alltoall/logs/"

rsync -avzP --cvs-exclude --delete --exclude="**~" "$remote" "$root/logs/"
