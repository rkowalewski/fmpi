#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J $jobname
#Output and error (also --output, --error):
#SBATCH -o $datadir/%j.n$nodes.p$ntasks.t$threads.out
#SBATCH -e $datadir/%j.n$nodes.p$ntasks.t$threads.err
#Initial working directory (also --chdir):
#SBATCH -D $directory
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=roger.kowalewski@ifi.lmu.de
# Wall clock limit:
#SBATCH --time=$time
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr92fo
# For benchmarking disable dynamic frequency scaling
#SBATCH --ear=off
# Job options
#SBATCH --partition $partition
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks

#Important
module load slurm_setup
module load hwloc

module unload gcc
module load gcc/9
module unload mpi.intel
module load mpi.intel/2019_gcc

unset KMP_AFFINITY

num_mgmt_threads="1"
num_threads_domain="$$(($threads + num_mgmt_threads))"

ncores="$$(getconf _NPROCESSORS_ONLN)"

ht_enabled="0"

if [[ "$$(($ntasks * num_threads_domain))" -gt "$$((ncores / 2))" ]]; then
  ht_enabled="1"
fi

mpiexec -n $$SLURM_NTASKS \
    -genv OMP_NUM_THREADS="$threads" \
    -genv FMPI_ENABLE_SMT="$$ht_enabled" \
    -genv FMPI_HW_CORES="$$ncores" \
    -genv FMPI_MGMT_CPUS="$$num_mgmt_threads" \
    ./jobs/impi_omp.sh \
    "build/benchmark/twoSidedAlgorithms.d" \
    $$SLURM_JOB_NUM_NODES $binary_args
