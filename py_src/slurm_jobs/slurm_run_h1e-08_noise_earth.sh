#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=run_h1e-08_noise_earth 
#SBATCH --output=outputs/run_h1e-08_noise_earth_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py run_h1e-08_noise_earth 1e-08 earth