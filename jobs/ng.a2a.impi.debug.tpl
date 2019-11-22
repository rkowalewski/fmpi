#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J <<JOBNAME>>.<<CONTEXT>>
#Output and error (also --output, --error):
#SBATCH -o <<LOGDIR>>/%j.n<<NUM_NODES>>.p<<NUM_PROCS>>.t<<NUM_THREADS>>.out
#SBATCH -e <<LOGDIR>>/%j.n<<NUM_NODES>>.p<<NUM_PROCS>>.t<<NUM_THREADS>>.err
#Initial working directory (also --chdir):
#SBATCH -D <<WORKDIR>>
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
#SBATCH --ntasks=<<NUM_TASKS>>
#SBATCH --nodes=<<NUM_NODES>>
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=<<NUM_PROCS>>

#Important
module load slurm_setup

export RTLX_ENABLE_TRACE=1

mpiexec \
    -n $((<<NUM_PROCS>> * <<NUM_NODES>>)) build/benchmark/twoSidedAlgorithms.d \
    $SLURM_JOB_NUM_NODES <<ARGS>>
