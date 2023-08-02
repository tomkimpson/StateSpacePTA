#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=noise_batch_1314 
#SBATCH --output=outputs/noise_batch_1314_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py noise_batch_1314 1e-12 earth 1314