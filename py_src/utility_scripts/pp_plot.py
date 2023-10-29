import numpy as np 
import sys






def calculate_quantiles(list_of_files):


    quantiles = np.arange(0,1.0,0.05) #[0.05,0.95] #90 per cent quantiles 
    quantiles = np.append(quantiles,1.0) #quantiles.append(1.0)
    quantile_results = np.zeros((int(len(quantiles)/2),len(variables_to_plot))) #quantile rows x variable columns


    for f in list_of_files:
        print(f)
        injection_parameters = parse_filename(f)
        # Load the data into a df
        f = open(f)
        data = json.load(f)
        f.close()
        df_posterior = pd.DataFrame(data["posterior"]["content"])[variables_to_plot] 
        
        #Get the quantiles
        df_quantile = df_posterior.quantile(quantiles)


        #For each quantile pair

        #Special case for i=0
        i=0
        lower_limit = df_quantile.iloc[i] 
        upper_limit = df_quantile.iloc[-1] 
        #print("quantiles", 0,quantiles[i], quantiles[-1],quantiles[-1] -quantiles[i])


        for j in range(len(injection_parameters)):
            var = variables_to_plot[j]
            #print(0, lower_limit[var], injection_parameters[j], upper_limit[var])
            if lower_limit[var] <= injection_parameters[j] <= upper_limit[var]:
                quantile_results[i,j] = quantile_results[i,j] + 1





        for i in range(1,int(len(quantiles)/2)):
            #print("quantiles", i,quantiles[i], quantiles[-i],quantiles[-i] -quantiles[i])

            lower_limit = df_quantile.iloc[i] 
            upper_limit = df_quantile.iloc[-i-1] #-1 since we are ignoreing the last element which was handled in the special case

            for j in range(len(injection_parameters)):
                var = variables_to_plot[j]
                if lower_limit[var] <= injection_parameters[j] <= upper_limit[var]:
                    quantile_results[i,j] = quantile_results[i,j] + 1



    np.save("PP_plot_results",quantile_results)

   
