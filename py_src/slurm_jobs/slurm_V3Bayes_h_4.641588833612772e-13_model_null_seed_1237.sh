#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V3Bayes_h_4.641588833612772e-13_model_null_seed_1237 
#SBATCH --output=outputs/V3Bayes_h_4.641588833612772e-13_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V3Bayes_h_4.641588833612772e-13_model_null_seed_1237 4.641588833612772e-13 null 1237