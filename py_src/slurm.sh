#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=72:00:00 
#SBATCH --job-name=pulsar_terms_test_h1e14 
#SBATCH --output=outputs/pulsar_terms_test_h1e14_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py pulsar_terms_test_h1e14 1e-14 pulsar 1237