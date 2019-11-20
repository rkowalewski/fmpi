#!/usr/bin/env python3

import argparse
import numpy as np
from math import log2,ceil
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

log2min = log2(args.minsize)
log2max = log2(args.maxsize)

bsizes = np.logspace(log2min, log2max, num=log2max -
                     log2min+1, base=2, dtype='int')

nr = args.nodes * args.procs

algos = ["AlltoAll", "ScatterPairwise", "Bruck"]

# commstate:
#    communication buffer: blocksize * nreqs * 2
#        -> occupation in average: nreqs
#    merge buffer: blocksize * nr, only once for final merge


def bruck(blocksize, nr):
    return blocksize * ceil(nr / 2)

def scatterPairwiseBuf(blocksize, nr, n, sendcount, nreqs):
    return blocksize * reqwin * 2



iterations = []

for i in range(len(bsizes)):
    blocksize = bsizes[i]

    sendcount = blocksize * nr
    recvcount = sendcount
    mergebuf = recvcount

    reqwin = 4

    d = {}
    d["nprocs"] = nr
    d["algorithm"] = "ScatterPairwise"
    d["blocksize"] = blocksize
    d["sendcount"] = sendcount
    d["recvcount"] = recvcount
    d["commbuf"] = scatterPairwiseBuf(blocksize, sendcount, reqwin)
    d["mergebuf"] = mergebuf
    d["total"] = d["commbuf"] + sendcount + recvcount + mergebuf
    d["total"] /= 1024 # calculate everything in KB
    iterations.append(d)

if len(iterations) > 1:
    keys = iterations[0].keys()
    dict_writer = csv.DictWriter(sys.stdout, keys)
    dict_writer.writeheader()
    dict_writer.writerows(iterations)
