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

. ./jobs/env.impi.gcc9.sh

unset KMP_AFFINITY

mpiexec -n $((<<NUM_PROCS>> * <<NUM_NODES>>)) \
    -genv OMP_NUM_THREADS=<<NUM_THREADS>> \
    -genv FMPI_ENABLE_SMT=0 \
    -genv FMPI_HW_CORES="$(getconf _NPROCESSORS_ONLN)" \
    ./jobs/impi_omp.sh "./build.release/apps/mpiPinning"

