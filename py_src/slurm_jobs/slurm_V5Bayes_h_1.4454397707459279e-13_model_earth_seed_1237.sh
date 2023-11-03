#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V5Bayes_h_1.4454397707459279e-13_model_earth_seed_1237 
#SBATCH --output=outputs/V5Bayes_h_1.4454397707459279e-13_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V5Bayes_h_1.4454397707459279e-13_model_earth_seed_1237 1.4454397707459279e-13 earth 1237