#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=batch_1236 
#SBATCH --output=outputs/batch_1236_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py batch_1236 1e-12 earth 1236