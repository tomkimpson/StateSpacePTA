#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=badger_cheap_fprior 
#SBATCH --output=outputs/badger_cheap_fprior_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py badger_cheap_fprior 1e-2 earth