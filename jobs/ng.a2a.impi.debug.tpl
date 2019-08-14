#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J <<JOB_NAME>>
#Output and error (also --output, --error):
#SBATCH -o /dss/dsshome1/03/%u/logs/%x/%j.n<<NUM_NODES>>.p<<NUM_PROCS>>.t<<NUM_THREADS>>.out
#SBATCH -e /dss/dsshome1/03/%u/logs/%x/%j.n<<NUM_NODES>>.p<<NUM_PROCS>>.t<<NUM_THREADS>>.err
#Initial working directory (also --chdir):
#SBATCH -D /dss/dsshome1/03/di25qoy2/workspaces/alltoall
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=roger.kowalewski@ifi.lmu.de
# Wall clock limit:
#SBATCH --time=<<WCLIMIT>>
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr92fo
# For benchmarking disable dynamic frequency scaling
#SBATCH --ear=off
# Job options
#SBATCH --partition=<<CLASS>>
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=<<NUM_NODES>>
#SBATCH --ntasks-per-node=<<NUM_PROCS>>
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1

#Important
module load slurm_setup

export A2A_ENABLE_TRACE=1

mpiexec -n $SLURM_NTASKS ./build.impi/MpiAlltoAllBench.d $SLURM_JOB_NUM_NODES