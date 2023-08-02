#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V4Bayes_h_7.943282347242822e-15_model_null_seed_1237 
#SBATCH --output=outputs/V4Bayes_h_7.943282347242822e-15_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V4Bayes_h_7.943282347242822e-15_model_null_seed_1237 7.943282347242822e-15 null 1237