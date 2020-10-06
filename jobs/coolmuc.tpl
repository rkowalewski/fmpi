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
# Job options
#SBATCH --clusters=$partition
# Wall clock limit:
#SBATCH --time=$time
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks
#SBATCH --export=NONE
#SBATCH --get-user-env


module load gcc/9
module load boost

numactl -H

unset KMP_AFFINITY

num_mgmt_threads="1"
num_threads_domain="$$(($threads + num_mgmt_threads))"

ncores="$$(getconf _NPROCESSORS_ONLN)"

ht_enabled="0"

if [[ "$$(($ntasks * num_threads_domain))" -gt "$$((ncores / 2))" ]]; then
  ht_enabled="1"
  echo "hyperthreading enabled"
fi

mpiexec -n $$SLURM_NTASKS \
    -genv OMP_NUM_THREADS="$threads" \
    -genv FMPI_ENABLE_SMT="$$ht_enabled" \
    -genv FMPI_HW_CORES="$$ncores" \
    -genv FMPI_MGMT_CPUS="$$num_mgmt_threads" \
    ./jobs/impi_omp.sh \
    "$binary" \
    $$SLURM_JOB_NUM_NODES $binary_args
