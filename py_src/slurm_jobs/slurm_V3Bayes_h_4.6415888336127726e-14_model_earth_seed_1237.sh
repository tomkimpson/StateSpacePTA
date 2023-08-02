#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=24:00:00 
#SBATCH --job-name=V3Bayes_h_4.6415888336127726e-14_model_earth_seed_1237 
#SBATCH --output=outputs/V3Bayes_h_4.6415888336127726e-14_model_earth_seed_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py V3Bayes_h_4.6415888336127726e-14_model_earth_seed_1237 4.6415888336127726e-14 earth 1237