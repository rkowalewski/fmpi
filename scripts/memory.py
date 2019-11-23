#!/usr/bin/env python3

import argparse
#import numpy as np
from math import log2, ceil, pow
import sys
import csv


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


parser = argparse.ArgumentParser(description='Calculate memory overhead.')
parser.add_argument('nodes', type=check_positive)
parser.add_argument('procs', type=check_positive)
parser.add_argument('minsize', type=check_positive)
parser.add_argument('maxsize', type=check_positive)

args = parser.parse_args()

log2min = int(log2(args.minsize))
log2max = int(log2(args.maxsize))

bsizes = []
for i in range(log2min, log2max + 1):
    bsizes.append(int(pow(2, i)))

print(bsizes)

sys.exit(0)

nr = args.nodes * args.procs

algos = ["AlltoAll", "ScatterPairwise", "Bruck"]

# commstate:
#    communication buffer: blocksize * nreqs * 2
#        -> occupation in average: nreqs
#    merge buffer: blocksize * nr, only once for final merge


def bruck(blocksize, nr):
    # we exchange ceil(nr / 2) blocks per round
    # but we need contiguous pack and unpack buffers
    return blocksize * nr


def scatterPairwiseBuf(blocksize, winsize):
    return blocksize * winsize * 2


iterations = []

algos = ["ScatterPairwise", "Bruck"]

for i in range(len(bsizes)):
    blocksize = bsizes[i]

    sendcount = blocksize * nr
    recvcount = sendcount
    mergebuf = recvcount

    reqwin = 8

    for algo in algos:
        d = {}
        d["nprocs"] = nr
        d["algorithm"] = "ScatterPairwise"
        d["blocksize"] = blocksize
        d["sendcount"] = sendcount
        d["recvcount"] = recvcount
        d["mergebuf"] = mergebuf

        if (algo == "ScatterPairwise"):
            d["commbuf"] = scatterPairwiseBuf(blocksize, reqwin)
        elif (algo == "Bruck"):
            d["commbuf"] = Bruck(blocksize, nr)

        d["totalKB"] = d["commbuf"] + sendcount + recvcount
        d["totalKB"] /= 1024  # calculate everything in KB

        iterations.append(d)

if len(iterations) > 1:
    keys = iterations[0].keys()
    dict_writer = csv.DictWriter(sys.stdout, keys)
    dict_writer.writeheader()
    dict_writer.writerows(iterations)
