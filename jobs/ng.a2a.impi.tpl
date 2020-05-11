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

. ./jobs/env.impi.gcc9.sh

unset KMP_AFFINITY

num_procs="<<NUM_PROCS>>"
num_nodes="<<NUM_NODES>>"
num_comp_threads="<<NUM_THREADS>>"

num_mgmt_threads="1"
num_threads_domain="$((num_comp_threads + num_mgmt_threads))"

ncores="$(getconf _NPROCESSORS_ONLN)"

ht_enabled="0"

if [[ "$((num_procs * num_threads_domain))" -gt "$((ncores / 2))" ]]; then
  ht_enabled="1"
fi

mpiexec -n $((num_procs * num_nodes)) \
    -genv OMP_NUM_THREADS="$num_comp_threads" \
    -genv FMPI_ENABLE_HT="$ht_enabled" \
    -genv FMPI_HW_CORES="$ncores" \
    ./jobs/impi_omp.sh \
    "build.release/benchmark/twoSidedAlgorithms" \
    <<ARGS>>
