#!/bin/bash

#SBATCH --account=nn8008k  # Substitute with your project name
#SBATCH --job-name=ecn_ensemble


#SBATCH --time=1-20:00:00
#SBATCH --nodes=1
# Safety settings
set -o errexit
set -o nounset
export JULIA_NUM_THREADS=16

# Load MPI module
module load Julia/1.7.1-linux-x86_64 



srun --ntasks=16 --cpus-per-task=1 --account=nn8008k --time=12:00:00 -o ecn.out -e ecn.errr julia ensemble_ecn.jl & 

wait
