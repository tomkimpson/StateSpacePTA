




import os
import sys 
import numpy as np 


def create_slurm_job(arg_name,h,measurement_model,seed):

    with open(f'../slurm_jobs/slurm_{arg_name}.sh','w') as g:


        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=24:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main.py {arg_name} {h} {measurement_model} {seed}")
    
    

# N = 1000
# seeds = np.arange(1235+10,1235+10+N,1)
# h = 5e-15 
# model = "earth"
# with open('batch.sh','w') as b: 

#     for s in seeds:
#         arg_name = f"SMALL_H_NOISE_model_earth_batch_{s}"
#         create_slurm_job(arg_name,h,model,s)
#         b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")
       

h_range = np.logspace(-15,-12,101)
noise_models = ["earth", "null"]
#noise_models = ["earth"]

seeds = np.arange(1245,1245+10)

with open('batch.sh','w') as b:
    
    for s in seeds:
        for h in h_range:
            for n in noise_models:
                 arg_name = f"KMULTIpaper_bayes_ratios_n2500_V1_h_{h}_model_{n}_seed_{s}"
                 print(arg_name)
                 create_slurm_job(arg_name,h,n,s)

                 b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")

    

