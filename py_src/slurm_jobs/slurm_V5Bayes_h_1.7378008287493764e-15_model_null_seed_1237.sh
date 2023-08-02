#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V5Bayes_h_1.7378008287493764e-15_model_null_seed_1237 
#SBATCH --output=outputs/V5Bayes_h_1.7378008287493764e-15_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V5Bayes_h_1.7378008287493764e-15_model_null_seed_1237 1.7378008287493764e-15 null 1237