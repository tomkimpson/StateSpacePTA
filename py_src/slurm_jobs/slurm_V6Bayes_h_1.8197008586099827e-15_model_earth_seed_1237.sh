#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V6Bayes_h_1.8197008586099827e-15_model_earth_seed_1237 
#SBATCH --output=outputs/V6Bayes_h_1.8197008586099827e-15_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V6Bayes_h_1.8197008586099827e-15_model_earth_seed_1237 1.8197008586099827e-15 earth 1237