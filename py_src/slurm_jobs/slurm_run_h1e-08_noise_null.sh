#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=run_h1e-08_noise_null 
#SBATCH --output=outputs/run_h1e-08_noise_null_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py run_h1e-08_noise_null 1e-08 null