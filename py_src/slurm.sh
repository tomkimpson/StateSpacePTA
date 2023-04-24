#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=mcmc_rerun 
#SBATCH --output=outputs/mcmc_rerun_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
python main.py mcmc_rerun