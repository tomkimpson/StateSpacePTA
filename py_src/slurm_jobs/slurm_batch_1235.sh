#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=batch_1235 
#SBATCH --output=outputs/batch_1235_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py batch_1235 1e-12 earth 1235