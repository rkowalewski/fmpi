BEGIN {
    OFS=","
}

NR == 2 {
    $1="Algorithm"
    $2="Time"
    $3="Iterations"
    $4="Procs"
    $5="Threads"
    $6="Blocksize"
    $7="Total"
    print
} # header

NR > 3 {
    split($1, bm, "/")
    sub(/BM_/,"", bm[1])
    print  bm[1] bm[4], $2 * 1E-9, $6, bm[2], bm[5], bm[3], bm[3] * bm[2]
} # measurements
