

#As config_batch.py but now using main_pp.py for generating PP plots


import os
import sys 
import numpy as np 
import bilby


def create_slurm_job(arg_name,h,measurement_model,seed,omega,phi0,psi,delta,alpha):

    with open(f'slurm_jobs/slurm_{arg_name}.sh','w') as g:


        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=24:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main_pp.py {arg_name} {h} {measurement_model} {seed} {omega} {phi0} {psi} {delta} {alpha}")
    
    

Nsamples = 100
seed = 1237 
h = 1e-12 

noise_models = ["earth", "null"]
        
omega_gw = 5e-7
phi0_gw = 0.20
psi_gw = 2.50 


#eps prevents railing against prior edge
eps =0.05
deltas = np.linspace(-np.pi/2.0+eps,np.pi/2.0 - eps,Nsamples)
alphas = np.linspace(0+eps,2*np.pi-eps,Nsamples)


with open('batch.sh','w') as b: 

    for m in noise_models:

        for d in deltas:
            for a in alphas:

   
                arg_name = f"skymap100_h_{h}_model_{m}_seed_{seed}_omega_{omega_gw}_phi0_{phi0_gw}_psi_{psi_gw}_delta_{d}_alpha_{a}"
                print(arg_name)
                create_slurm_job(arg_name,h,m,seed,omega_gw,phi0_gw,psi_gw,d,a)
                b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")
       