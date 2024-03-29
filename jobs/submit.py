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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_options(args, cwd):
    if not os.path.exists(args.binary):
        print("invalid path to binary: %s" % args.binary)
        return None

    canonical_path = os.path.abspath(args.binary)

    if not (stat.S_IXUSR & os.stat(canonical_path)[stat.ST_MODE]):
        print("binary not executable", args.binary)
        return None

    if args.jobname is None:
        args.jobname = os.path.basename(canonical_path)

    scratch = os.environ.get('SCRATCH')

    if scratch is None:
        scratch = os.path.join(cwd, '.logs')
    else:
        scratch = os.path.join(scratch, 'logs')

    datadir = "{scratch}/{project}/{job}".format(
        scratch=scratch, project="fmpi", job=args.jobname)

    if args.dryrun == False:
        mkdir_p(datadir)

    if args.binary_args is None:
        args.binary_args = ""

    options = {}

    partitionspec = "--partition {partition} ".format(partition=args.partition)

    if (args.cluster):
        partitionspec = partitionspec + "--clusters {cluster} ".format(cluster=args.cluster)

    if (args.qos):
        partitionspec = partitionspec + "--qos {qos} ".format(qos=args.qos)


    options['binary'] = args.binary
    options['nodes'] = args.nodes
    options['ntasks'] = args.ntasks
    options['threads'] = args.threads
    options['clusters'] = args.cluster
    options['partitionspec'] = partitionspec
    options['time'] = args.time
    options['jobname'] = args.jobname
    options['directory'] = cwd
    options['datadir'] = datadir
    options['binary_args'] = args.binary_args

    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Job submission.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('template', help='the template file', type=argparse.FileType('r'))
    parser.add_argument('binary', help='the binary to execute via mpiexec')
    parser.add_argument('--nodes', type=int, required=True,
                        help='number of nodes')
    parser.add_argument('--ntasks', type=int, required=True,
                        help='number of tasks per node')
    parser.add_argument('--threads', type=int, required=True,
                        help='number of threads per task')
    parser.add_argument('--time', help='wallclock time for job', required=True)
    parser.add_argument('--partition', help='slurm partition', required=True)
    parser.add_argument('--cluster', help='slurm cluster', required=False, default="")
    parser.add_argument('--qos', help='quality of service', required=False, default="")
    parser.add_argument(
        '--jobname', help='job name (if empty, the basename of the binary is used)')
    parser.add_argument(
        '--binary-args', help='arguments passed to the binary via mpirun')
    parser.add_argument('-n', '--dry-run', dest='dryrun', type=str2bool,
                        nargs='?', default=False, const=True, help="Dry run option")
    args = parser.parse_args()

    cwd = getGitRoot()
    options = process_options(args, cwd)
    if not (options is None):
        src = Template(args.template.read())
        result = src.substitute(options)

        print(result)  # output generated batch file
