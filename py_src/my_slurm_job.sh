#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=8000MB
#SBATCH --time=48:00:00
#SBATCH --job-name=NestedSampling
#SBATCH --output=likelihood_expt.txt

source ~/.bashrc
conda activate OzStar
python main.py
