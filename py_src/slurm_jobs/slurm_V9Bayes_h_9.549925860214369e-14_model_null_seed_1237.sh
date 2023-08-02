#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V9Bayes_h_9.549925860214369e-14_model_null_seed_1237 
#SBATCH --output=outputs/V9Bayes_h_9.549925860214369e-14_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V9Bayes_h_9.549925860214369e-14_model_null_seed_1237 9.549925860214369e-14 null 1237