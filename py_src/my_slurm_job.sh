#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=4000MB
#SBATCH --time=24:00:00
#SBATCH --job-name=NestedSampling
#SBATCH --output=slurm_output.txt

source ~/.bashrc
conda activate OzStar
python main.py
