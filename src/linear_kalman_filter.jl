


"""
Given some data recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
                     PTA::Pulsars,
                     parameters::GuessedParameters,
                     model::Symbol
                      ) where {NF<:AbstractFloat}

   

    @info "Running the Kalman filter for the measurement model defined via: ", model

    @unpack q,dt,t = PTA         #Get the parameters related to the PTA 
    @unpack γ,σp,σm = parameters #Get some of the parameters that we want to infer 

    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    N = size(observations)[2]     # number of timesteps
    @info "Size of the state space is: ", L 
    @info "Number of observations is : ", N



    #Initialise x and P
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = I(L) * σm*1e9 
    #println(σm)
    #P = I(L) * 1e-15

    #tmp_sigma = NF(1e-3)
    #P = diagm(fill(tmp_sigma ,L)) #maybe we want the uncertainty in the frequencies and the uncertainty in the parameters to be different?
   

    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,σp,dt)
    


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
    x,P = update(x,P, observations[:,1],t[1],parameters,q,R)
    x_results[1,:] = x  
    for i=2:N


        observation = observations[:,i]
        ti = t[i]
        x_predict,P_predict = predict(x,P,parameters,ti,dt,Q)
        x,P                 = update(x_predict,P_predict, observation,ti,parameters,q,R)

        x_results[i,:] = x 
         
      
     end 

    return x_results, likelihood


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


    #Pnew = I_KH*P

    return xnew, Pnew 
end 


function predict(x,P,parameters,t,dt,Q)

    
    F = F_function(parameters,dt)
    T = T_function(parameters,t,dt)

    xp = F*x .+ T 

    Pp = F*P*F + Q

    return xp,Pp

end 



