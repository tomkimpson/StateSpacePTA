#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V5Bayes_h_4.466835921509635e-14_model_null_seed_1237 
#SBATCH --output=outputs/V5Bayes_h_4.466835921509635e-14_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V5Bayes_h_4.466835921509635e-14_model_null_seed_1237 4.466835921509635e-14 null 1237