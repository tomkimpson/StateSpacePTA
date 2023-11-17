

import json 
import pandas as pd 




def process_file(path,savepath):

    # Opening JSON file
    f = open(path)
    data = json.load(f)
    evidence = data["log_evidence"]
    f.close()

    df_posterior = pd.DataFrame(data["posterior"]["content"]) # make it a df
    
    #Just select some variables. We will get GW params and chi
    gw_parameters = ["omega_gw","phi0_gw","psi_gw","iota_gw","delta_gw","alpha_gw", "h"]
    chi_parameters = [f'chi{i}' for i in range(47)]
    variables_to_plot = gw_parameters + chi_parameters
       
    #Cropped df 
    df_cropped = df_posterior[variables_to_plot]

    #add the evidence. The same for every sample of course
    df_cropped['evidence'] = evidence

    #save to disk
    df_cropped.to_parquet(df_cropped)


