#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V8Bayes_h_5.1286138399136585e-14_model_null_seed_1237 
#SBATCH --output=outputs/V8Bayes_h_5.1286138399136585e-14_model_null_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V8Bayes_h_5.1286138399136585e-14_model_null_seed_1237 5.1286138399136585e-14 null 1237