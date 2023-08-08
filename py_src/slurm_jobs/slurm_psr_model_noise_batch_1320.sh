#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=psr_model_noise_batch_1320 
#SBATCH --output=outputs/psr_model_noise_batch_1320_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py psr_model_noise_batch_1320 1e-12 pulsar 1320