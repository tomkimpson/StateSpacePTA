


"""
Given some data recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
                     PTA::Pulsars,
                     parameters::GuessedParameters,
                     model::Symbol
                      ) where {NF<:AbstractFloat}

   

    @info "Running the Kalman filter for the measurement model defined via: ", model

    @unpack q,dt,t = PTA # PTA does have some parameters that ultimatley we might want to estimate . 

    @unpack γ,σp,σm = parameters 

    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    
    
    N = size(observations)[2]     # number of timesteps
    @info "Size of the state space is: ", L 
    @info "Number of observations is : ", N


    #Get the 0th order frequencies and reshape them 
    #These are the PSR frequencies given by ANTF
    #f0 = reshape(f0,(1,size(f0)[1])) #change f0 from a 1D vector to a 2D matrix

    #Initialise x and P
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    #x_GW_parameters = [parameters.h, parameters.ι, parameters.δ, parameters.α, parameters.ψ, parameters.ω, parameters.Φ0] # guess the GW parameters 


   # x = [x_pulsar_frequencies; x_GW_parameters] #concatenate to get the intial state
    #P = I(L) * σm*1e9 
    
    tmp_sigma = NF(1e-3)
    P = diagm(fill(tmp_sigma ,L)) #maybe we want the uncertainty in the frequencies and the uncertainty in the parameters to be different?
    # #Initialise the weights for thexite UKF


    #Calculate the time-independent Q-matrix
    Q_function(γ,σp,dt)
    


    #Calculate measurement noise matrix
    R = R_function(Npulsars,σm)

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

    #Update step first, then iterate over remainders
    #See e.g. https://github.com/meyers-academic/baboo/blob/f5619df23b2465373443e02cd52e1003ed66c0ac/baboo/kalman.py#L167
    x,P = update(x,P, observation[:,i],t[i],parameters,q,R) 
    for i=2:1 #N

        println("STEP NUMBER i ", i)
        #Grab the observatin and the time 
        observation = observations[:,i]
         #observation = reshape(observation,(1,size(observation)[1])) #can we avoid these reshapes and get a consistent dimensionality?
         ti = t[i]


        x1,P1 = update(x,P, observation,ti,parameters,q,R)
        #  self.update(obs)    # Kalman update step
        #  self.predict()      # Kalman predict step
         
      
     end 

    return x_results, likelihood
    #return 1,1

end 




function update(x,P, observation,t,parameters,q,R)

    H = H_function(parameters,t,q)
    y = observation .- H*x 
    S = H*P*H .+ R 
    K = P*H*inv(S)
    xnew = x .+ K*y

    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = I - (K*H)
    Pnew = I_KH * P * I_KH' .+ K * R * K'

    return xnew, Pnew 
end 


function predict(x,p)


    xp = 



end 



