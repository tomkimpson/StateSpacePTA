#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V4Bayes_h_5.011872336272715e-13_model_earth_seed_1237 
#SBATCH --output=outputs/V4Bayes_h_5.011872336272715e-13_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V4Bayes_h_5.011872336272715e-13_model_earth_seed_1237 5.011872336272715e-13 earth 1237