#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=exponential_form_dynesty 
#SBATCH --output=outputs/exponential_form_dynesty_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
python main.py exponential_form_dynesty