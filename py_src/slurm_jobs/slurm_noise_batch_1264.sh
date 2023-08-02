#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=noise_batch_1264 
#SBATCH --output=outputs/noise_batch_1264_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py noise_batch_1264 1e-12 earth 1264