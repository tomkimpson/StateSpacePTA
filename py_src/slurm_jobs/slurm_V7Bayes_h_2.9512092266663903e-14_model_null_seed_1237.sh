#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V7Bayes_h_2.9512092266663903e-14_model_null_seed_1237 
#SBATCH --output=outputs/V7Bayes_h_2.9512092266663903e-14_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V7Bayes_h_2.9512092266663903e-14_model_null_seed_1237 2.9512092266663903e-14 null 1237