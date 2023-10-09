




import numpy as np 


def create_slurm_job(arg_name,h,measurement_model,seed):

    with open(f'../slurm_jobs/slurm_{arg_name}.sh','w') as g:


        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=96:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main.py {arg_name} {h} {measurement_model} {seed}")
    

#BAYES PLOT
h_range = np.logspace(-15,-12,101)
noise_models = ["pulsar","earth", "null"]

s = 1250 #seed. Also try 1245 which I think is what was used in the paper: https://github.com/tomkimpson/StateSpacePTA/blob/9d997dc7d42ae612e7d526d34b0661944af6eb99/py_src/config_batch.py
with open('batch.sh','w') as b:
    
    for h in h_range:
        for n in noise_models:
            arg_name = f"beta_canonical_bayes_h_{h}_model_{n}"
            print(arg_name)
            create_slurm_job(arg_name,h,n,s)

            b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")



