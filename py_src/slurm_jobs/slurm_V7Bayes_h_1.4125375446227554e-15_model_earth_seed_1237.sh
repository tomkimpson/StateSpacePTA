#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V7Bayes_h_1.4125375446227554e-15_model_earth_seed_1237 
#SBATCH --output=outputs/V7Bayes_h_1.4125375446227554e-15_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V7Bayes_h_1.4125375446227554e-15_model_earth_seed_1237 1.4125375446227554e-15 earth 1237