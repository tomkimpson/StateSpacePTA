#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V7Bayes_h_2.2387211385683377e-14_model_earth_seed_1237 
#SBATCH --output=outputs/V7Bayes_h_2.2387211385683377e-14_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V7Bayes_h_2.2387211385683377e-14_model_earth_seed_1237 2.2387211385683377e-14 earth 1237