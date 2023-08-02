#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V8Bayes_h_7.079457843841374e-13_model_null_seed_1237 
#SBATCH --output=outputs/V8Bayes_h_7.079457843841374e-13_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V8Bayes_h_7.079457843841374e-13_model_null_seed_1237 7.079457843841374e-13 null 1237