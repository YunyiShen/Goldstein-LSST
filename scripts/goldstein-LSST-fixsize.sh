#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-10:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=8000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o goldstein-LSST-fixsize_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e goldstein-LSST-fixsize_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu

# load modules
module load python/3.10.13-fasrc01
mamba activate jax

python goldstein-LSST-fixsize.py
