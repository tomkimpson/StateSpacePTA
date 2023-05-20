#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=batch_1237 
#SBATCH --output=outputs/batch_1237_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py batch_1237 1e-12 earth 1237