#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V6Bayes_h_2.187761623949552e-15_model_earth_seed_1237 
#SBATCH --output=outputs/V6Bayes_h_2.187761623949552e-15_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V6Bayes_h_2.187761623949552e-15_model_earth_seed_1237 2.187761623949552e-15 earth 1237