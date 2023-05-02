


### A python script that accepts a name argument and populates a slurm file 


import os
import sys 


arg_name = sys.argv[1]

with open('slurm.sh','w') as g:


    g.write("#!/bin/bash \n \n")  
    g.write("#SBATCH --ntasks=1 \n")  
    g.write("#SBATCH --mem=8000MB \n")  
    g.write("#SBATCH --time=48:00:00 \n")  
    g.write(f"#SBATCH --job-name={arg_name} \n")  
    g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

    g.write("source ~/.bashrc \n")
    g.write("conda activate OzStar \n")
    g.write(f"time python main.py {arg_name}")
    
    
    

