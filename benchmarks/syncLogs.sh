#!/bin/sh

rsync -avzP --cvs-exclude --delete --exclude="**~" "skx.supermuc:~/logs/ng.a2a.impi" /media/kowalewski/benchmarks/logs/
