#!/usr/bin/env python3

import pathlib
import os
import stat
import subprocess
import argparse
import pathlib
from string import Template


def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    try:
        pathlib.Path(dir).mkdir(parents=True)
    except FileExistsError:
        pass

parser = argparse.ArgumentParser(description='Job submission.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                 )
parser.add_argument('binary', help='the binary to execute via mpiexec')

parser.add_argument('nodes', type=int,
                    help='number of nodes')
parser.add_argument('ntasks', type=int,
                    help='number of tasks per node')
parser.add_argument('threads', type=int,
                    help='number of threads per task')
parser.add_argument(
    '--jobname', help='job name (if empty, the basename of the binary is used)')
parser.add_argument('--partition', help='slurm partition', default='test')
parser.add_argument('--time', help='wallclock time for job', default='00:30:00')
parser.add_argument(
    '--binary-args', help='arguments passed to the binary via mpirun')


args = parser.parse_args()

if not os.path.exists(args.binary):
    print("invalid path to binary: %s" % args.binary)
    exit(1)

canonical_path = os.path.abspath(args.binary)

if not (stat.S_IXUSR & os.stat(canonical_path)[stat.ST_MODE]):
    print("binary not executable", args.binary)
    exit(1)

if args.jobname is None:
    args.jobname = os.path.basename(canonical_path)

cwd = getGitRoot()

scratch = os.environ.get('SCRATCH')

if scratch is None:
    scratch = os.path.join(cwd, '.logs')
else:
    scratch = os.path.join(scratch, 'logs')

datadir = "{scratch}/{project}/{job}".format(
    scratch=scratch, project="fmpi", job=args.jobname)

mkdir_p(datadir)

if args.binary_args is None:
    args.binary_args = ""

options = {}

options['nodes'] = args.nodes
options['ntasks'] = args.ntasks
options['threads'] = args.threads
options['partition'] = args.partition
options['time'] = args.time
options['jobname'] = args.jobname
options['directory'] = cwd
options['datadir'] = datadir
options['binary_args'] = args.binary_args


filein = open(os.path.join(cwd, 'jobs/sbatch.tpl'))

src = Template(filein.read())
result = src.substitute(options)

print(result)  # output generated batch file
