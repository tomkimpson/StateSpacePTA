#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V7Bayes_h_6.4565422903465625e-15_model_null_seed_1237 
#SBATCH --output=outputs/V7Bayes_h_6.4565422903465625e-15_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V7Bayes_h_6.4565422903465625e-15_model_null_seed_1237 6.4565422903465625e-15 null 1237