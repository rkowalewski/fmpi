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
module load hwloc

unset KMP_AFFINITY

#export OMP_NUM_THREADS=<<NUM_THREADS>>

export FMPI_DOMAIN_SIZE="$((48 / <<NUM_PROCS>>))"
export OMP_NUM_THREADS="$((FMPI_DOMAIN_SIZE - 1))"
export OMP_PROC_BIND="true"
export OMP_PLACES="$(./scripts/genPlaces.sh <<NUM_PROCS>>)"


mpiexec \
    -env I_MPI_PIN_DOMAIN $((96 / <<NUM_PROCS>>)) \
    -env I_MPI_PIN_ORDER compact \
    -n $SLURM_NTASKS  \
    build/benchmark/twoSidedAlgorithms.d \
    $SLURM_JOB_NUM_NODES <<ARGS>>
