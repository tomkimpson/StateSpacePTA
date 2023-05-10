#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=test1 
#SBATCH --output=outputs/test1_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py test1 1e-2 True