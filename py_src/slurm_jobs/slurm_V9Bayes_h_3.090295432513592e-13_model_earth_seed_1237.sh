#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V9Bayes_h_3.090295432513592e-13_model_earth_seed_1237 
#SBATCH --output=outputs/V9Bayes_h_3.090295432513592e-13_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V9Bayes_h_3.090295432513592e-13_model_earth_seed_1237 3.090295432513592e-13 earth 1237