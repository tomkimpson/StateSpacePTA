

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
        g.write("#SBATCH --time=48:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main_pp.py {arg_name} {h} {measurement_model} {seed} {omega} {phi0} {psi} {delta} {alpha}")
    
    

N = 200
seed = 1250 
h = 5e-15 
model = "pulsar"
        
#Create a bilby prior object to sample from
init_parameters = {}
priors = bilby.core.prior.PriorDict()

init_parameters["omega_gw"] = None
priors["omega_gw"] = bilby.core.prior.LogUniform(1e-8, 1e-6, 'omega_gw')

init_parameters["phi0_gw"] = None
priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')

init_parameters["psi_gw"] = None
priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

init_parameters["delta_gw"] = None
priors["delta_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')

init_parameters["alpha_gw"] = None
priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')


with open('batch.sh','w') as b: 

    for i in range(N):
        p = priors.sample()

        arg_name = f"N200chi_single_pp_plot_h_{h}_model_{model}_seed_{seed}_omega_{p['omega_gw']}_phi0_{p['phi0_gw']}_psi_{p['psi_gw']}_delta_{p['delta_gw']}_alpha_{p['alpha_gw']}"
        print(arg_name)
        create_slurm_job(arg_name,h,model,seed,p['omega_gw'],p['phi0_gw'],p['psi_gw'],p['delta_gw'],p['alpha_gw'])
        b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")
       


