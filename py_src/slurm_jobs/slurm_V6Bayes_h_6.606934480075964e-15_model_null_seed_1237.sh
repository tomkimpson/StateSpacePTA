#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V6Bayes_h_6.606934480075964e-15_model_null_seed_1237 
#SBATCH --output=outputs/V6Bayes_h_6.606934480075964e-15_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V6Bayes_h_6.606934480075964e-15_model_null_seed_1237 6.606934480075964e-15 null 1237