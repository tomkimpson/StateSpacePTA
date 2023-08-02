#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V8Bayes_h_1.9054607179632443e-13_model_earth_seed_1237 
#SBATCH --output=outputs/V8Bayes_h_1.9054607179632443e-13_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V8Bayes_h_1.9054607179632443e-13_model_earth_seed_1237 1.9054607179632443e-13 earth 1237