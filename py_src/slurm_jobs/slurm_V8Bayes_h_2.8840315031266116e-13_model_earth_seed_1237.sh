#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V8Bayes_h_2.8840315031266116e-13_model_earth_seed_1237 
#SBATCH --output=outputs/V8Bayes_h_2.8840315031266116e-13_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V8Bayes_h_2.8840315031266116e-13_model_earth_seed_1237 2.8840315031266116e-13 earth 1237