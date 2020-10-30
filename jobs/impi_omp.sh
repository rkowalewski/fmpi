#!/bin/bash

print_stderr() {
  cat <<< "$PMI_RANK: $@" 1>&2
}

my_command="$1"
shift

ncpus="$((FMPI_HW_CORES / 2))"

domain_size="$((ncpus / SLURM_NTASKS_PER_NODE))"
comm_cpus="$FMPI_MGMT_CPUS"

# print_stderr "ncpus=$ncpus"
# print_stderr "domain_size=$domain_size"
# print_stderr "comm_cpus=$comm_cpus"

offset="$comm_cpus"

global_rank="$PMI_RANK"
local_rank=$((global_rank % SLURM_NTASKS_PER_NODE))

omp_fst_place="$((local_rank * domain_size + offset))"

if [[ "$FMPI_ENABLE_SMT" -eq "0" ]]
then
  omp_snd_place="$((omp_fst_place + ncpus))"
  omp_stride="1"
  omp_place_spec="{$omp_fst_place,$omp_snd_place}:$OMP_NUM_THREADS:$omp_stride"
else
  omp_snd_place="$((omp_fst_place + (OMP_NUM_THREADS / 2) - 1))"

  omp_place_spec=""

  for p in $(seq $omp_fst_place $omp_snd_place); do
    omp_place_spec="$omp_place_spec,{$p},{$((p+$ncpus))}"
  done
  # remove first comma
  omp_place_spec="${omp_place_spec#?}"
fi

# Variables for FMPI

export OMP_PLACES="$omp_place_spec"
# prevent any thread migration
export OMP_PROC_BIND="true"
export FMPI_DOMAIN_SIZE="$domain_size"

print_stderr "${OMP_PLACES}"

print_stderr "$my_command" "$@"

"$my_command" "$@"
