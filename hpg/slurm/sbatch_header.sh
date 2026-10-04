#!/bin/bash
# Shared #SBATCH defaults for MMCTL experiments on hpg-turin.
# Reference only — SLURM reads directives from the submitted .slurm file, so these
# are copied into each one.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=72:00:00
#SBATCH --mem=50gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhou.m@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
