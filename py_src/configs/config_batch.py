




import numpy as np 


def create_slurm_job(arg_name,h,measurement_model,seed,num_gw_sources):

    with open(f'../slurm_jobs/slurm_{arg_name}.sh','w') as g:


        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=96:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main.py {arg_name} {h} {measurement_model} {seed} {num_gw_sources}")
    

h = 5e-15
model = 'pulsar'
seed = 1237
nums = [1,2,5,10,20]

with open('batch.sh','w') as b:
    
    for n in nums:
        arg_name = f"HD_revisited_num_gw_{n}"
        print(arg_name)
        create_slurm_job(arg_name,h,model,seed,n)
        b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")



