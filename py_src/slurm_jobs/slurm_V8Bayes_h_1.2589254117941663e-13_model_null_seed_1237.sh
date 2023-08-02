#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V8Bayes_h_1.2589254117941663e-13_model_null_seed_1237 
#SBATCH --output=outputs/V8Bayes_h_1.2589254117941663e-13_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V8Bayes_h_1.2589254117941663e-13_model_null_seed_1237 1.2589254117941663e-13 null 1237