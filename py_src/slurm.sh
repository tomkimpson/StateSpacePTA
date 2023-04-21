#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=MCMC_small_h_5000_v2 
#SBATCH --output=outputs/MCMC_small_h_5000_v2_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
python main.py MCMC_small_h_5000_v2