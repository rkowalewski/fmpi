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
#SBATCH --nodes=<<NUM_NODES>>
#SBATCH --ntasks-per-node=<<NUM_PROCS>>

#Important
module load slurm_setup

# Note: We do not need any additional pinning here.
# Intel MPI is smart enough and can handle both hyperthreading and
# non hyperthreading cases here

numactl -H

unset KMP_AFFINITY

export OMP_NUM_THREADS="<<NUM_THREADS>>"
export OMP_PLACES="{0:<<NUM_THREADS>>:1}"

./build.release/apps/mergebench