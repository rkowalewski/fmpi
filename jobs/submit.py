#!/usr/bin/env python3

import os,stat
import subprocess
import argparse

def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


parser = argparse.ArgumentParser(description='Job submission.')
parser.add_argument('nodes', type=int,
                            help='number of nodes')
parser.add_argument('ntasks', type=int,
                            help='number of tasks per node')
parser.add_argument('threads', type=int,
                            help='number of threads per task')
parser.add_argument('application', help='the binary to execute via mpirun')
parser.add_argument("jobname", help="jobname")
parser.add_argument("--binary-args", help="arguments passed to the binary via mpirun")
parser.add_argument("--class", help="job class", default="test")
parser.add_argument("--time", help="wallclock time for job")

args = parser.parse_args()

if not os.path.exists(args.application):
    print("invalid path to binary: %s" % args.application)
    exit(1)

canonical_path=os.path.abspath(args.application)

if not (stat.S_IXUSR & os.stat(canonical_path)[stat.ST_MODE]):
    print("binary not executable", args.application)
    exit(1)

if not args.jobname:
    args.jobname = os.path.basename(canonical_path)

root=getGitRoot()

scratch = os.environ['SCRATCH']
data_dir="{root}/logs/{project}/{job}".format(root=scratch, project="fmpi", job=args.jobname)

print(data_dir)

# # Make top level directories
# mkdir_p(job_directory)
# mkdir_p(data_dir)
#
# lizards=["LizardA","LizardB"]
#
# for lizard in lizards:
#
#     job_file = os.path.join(job_directory,"%s.job" %lizard)
#     lizard_data = os.path.join(data_dir, lizard)
#
#     # Create lizard directories
#     mkdir_p(lizard_data)
#
#     with open(job_file) as fh:
#         fh.writelines("#!/bin/bash\n")
#         fh.writelines("#SBATCH --job-name=%s.job\n" % lizard)
#         fh.writelines("#SBATCH --output=.out/%s.out\n" % lizard)
#         fh.writelines("#SBATCH --error=.out/%s.err\n" % lizard)
#         fh.writelines("#SBATCH --time=2-00:00\n")
#         fh.writelines("#SBATCH --mem=12000\n")
#         fh.writelines("#SBATCH --qos=normal\n")
#         fh.writelines("#SBATCH --mail-type=ALL\n")
#         fh.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")
#         fh.writelines("Rscript $HOME/project/LizardLips/run.R %s potato shiabato\n" %lizard_data)
#
#     os.system("sbatch %s" %job_file)
