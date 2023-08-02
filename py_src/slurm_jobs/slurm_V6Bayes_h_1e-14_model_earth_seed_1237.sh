#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V6Bayes_h_1e-14_model_earth_seed_1237 
#SBATCH --output=outputs/V6Bayes_h_1e-14_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V6Bayes_h_1e-14_model_earth_seed_1237 1e-14 earth 1237