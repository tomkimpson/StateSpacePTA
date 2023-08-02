#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V7Bayes_h_4.677351412871981e-13_model_null_seed_1237 
#SBATCH --output=outputs/V7Bayes_h_4.677351412871981e-13_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V7Bayes_h_4.677351412871981e-13_model_null_seed_1237 4.677351412871981e-13 null 1237