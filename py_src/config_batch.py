




import os
import sys 
import numpy as np 


def create_slurm_job(arg_name,h,measurement_model):

    with open(f'slurm_jobs/slurm_{arg_name}.sh','w') as g:


        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=48:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main.py {arg_name} {h} {measurement_model}")
    
    



h_range = np.logspace(-9,-8,2)
noise_models = ["earth", "null"]

with open('batch.sh','w') as b:

    
    for h in h_range:
        for n in noise_models:

            arg_name = f"run_h{h}_noise_{n}"
            print(arg_name)
            create_slurm_job(arg_name,h,n)


            b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")

    

