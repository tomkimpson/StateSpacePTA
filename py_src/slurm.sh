#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=72:00:00 
#SBATCH --job-name=paper_canonical_example_earth_terms_1237_v2 
#SBATCH --output=outputs/paper_canonical_example_earth_terms_1237_v2_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py paper_canonical_example_earth_terms_1237_v2 1e-12 earth 1237