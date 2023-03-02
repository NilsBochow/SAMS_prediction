#!/bin/bash

#SBATCH --account=nn8008k  # Substitute with your project name
#SBATCH --job-name=ecn_ensemble
##SBATCH --qos=devel

#SBATCH --time=1-20:00:00
##SBATCH --time=00:30:00
#SBATCH --nodes=1
# Safety settings
set -o errexit
set -o nounset
export JULIA_NUM_THREADS=32

# Load MPI module
module load Julia/1.7.1-linux-x86_64 



srun --account=nn8008k julia ensemble_ecn.jl &> ecn.out & 

wait
