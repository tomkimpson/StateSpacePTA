


"""
Given some data recover the state and determine the likelihood
"""
function EKF(observations::Matrix{NF},
                       PTA::Pulsars,
                       parameters::GuessedParameters,
                       model::Symbol
                      ) where {NF<:AbstractFloat}

   

    @info "Running the Kalman filter for the measurement model defined via: ", model

    @unpack q,dt,t,f0 = PTA # PTA does have some parameters that ultimatley we might want to estimate . 

    #@unpack σm,σp,γ,ω,Φ0 = parameters 



    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars + 7
    #L = 6*Npulsars + 7 # dimension of hidden states i.e. number of pulsars, which have 5 parameters + a frequency state and the 7 GW parameters 
    
    
    N = size(observations)[2]     # number of timesteps
    @info "Size of the state space is: ", L 
    @info "Number of observations is : ", N


    #Get the 0th order frequencies and reshape them 
    #These are the PSR frequencies given by ANTF
    #f0 = reshape(f0,(1,size(f0)[1])) #change f0 from a 1D vector to a 2D matrix

    #Initialise x and P
    x_pulsar_frequencies = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    x_GW_parameters = [parameters.h, parameters.ι, parameters.δ, parameters.α, parameters.ψ, parameters.ω, parameters.Φ0] # guess the GW parameters 


    x = [x_pulsar_frequencies; x_GW_parameters] #concatenate to get the intial state
    #P = I(L) * σm*1e9 
    
    tmp_sigma = NF(1e-3)
    P = diagm(fill(tmp_sigma ,L)) #maybe we want the uncertainty in the frequencies and the uncertainty in the parameters to be different?
    # #Initialise the weights for thexite UKF
    ukf_weights = calculate_UKF_weights(NF,L,7e-4,2.0,0.0) #standard UKF parameters. Maybe make this arguments to be set?


    #Calculate the time-independent Q-matrix
    Q_function(PTA.γ,PTA.σp,dt)
    


    #Calculate measurement noise matrix
    R = R_function(Npulsars,PTA.σm)

    #Initialise an array to hold the results
    x_results = zeros(NF,N,L)

    #Initialise a likelihood variable
    likelihood = NF(0.0)

    #Set what measurement model to use
    if model == :GW
        measurement_function = H_function
    elseif model == :null
        measurement_function = null_function
    else
        println("Model is not defined. Choose one of :GW or :null" )
        return
    end 


    filter_parameters = Dict("q" => PTA.q, "d" => PTA.d)

     for i=1:1 #N

        println("STEP NUMBER i ", i)
         #Grab the observatin and the time 
         observation = observations[:,i]
         observation = reshape(observation,(1,size(observation)[1])) #can we avoid these reshapes and get a consistent dimensionality?
         ti = t[i]


        update(x, observation,ti,filter_parameters,Npulsars)
        #  self.update(obs)    # Kalman update step
        #  self.predict()      # Kalman predict step
         
      
     end 

    return x_results, likelihood
    #return 1,1

end 




function update(state, observation,t,parameters,Npulsars)

    println("this is the update function")
    

    hx = H_function(state,parameters,t,Npulsars)
    
    y = observation .- hx 

    H 


end 





