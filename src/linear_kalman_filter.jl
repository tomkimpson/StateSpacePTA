


"""
Given some data recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
             PTA::Pulsars,
             parameters::GuessedParameters,
                     model::Symbol
                      ) where {NF<:AbstractFloat}

   

    println("******************************************************************")
    println("******************************************************************")
    println("******************************************************************")
    println("******************************************************************")


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
    #P = I(L) * σm*1e9 
    P = I(L) * 1e-6*1e9 

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
    #println(1, "  ", x[1], "  ", x[2]," ", x[3])
    #println(1, "  ", P[1,1], "  ", P[2,2]," ", P[3,3])
    
    x,P = update(x,P, observations[:,1],t[1],parameters,q,R)
    #x_results[1,:] = x  
    for i=2:N

       # println(i, "  ", x[1], "  ", x[2]," ", x[3])
       # println(i, "  ", P[1,1], "  ", P[2,2]," ", P[3,3])


        observation = observations[:,i]
        #println(observation[1], "  ", observation[2]," ", observation[3])
        ti = t[i]
        x_predict,P_predict   = predict(x,P,parameters,ti,dt,Q)
        x,P,l                 = update(x_predict,P_predict, observation,ti,parameters,q,R)

        likelihood +=l
        #x_results[i,:] = x 
         
      
     end 

    #return x_results, likelihood
    return likelihood



end 



"""
Given some data recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
             PTA::Pulsars,
             parameters::GuessedParameters,
             ω_tuple::Vector{Float64},
             model::Symbol
             ) where {NF<:AbstractFloat}



    ω = ω_tuple[1]
    @unpack q,dt,t = PTA         #Get the parameters related to the PTA 
    @unpack γ,σp,σm = parameters #Get some of the parameters that we want to infer 

    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    N = size(observations)[2]     # number of timesteps
  



    #Initialise x and P
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = I(L) * 1e-6*1e9  #P = I(L) * σm*1e9 
    println(size(P))

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

    
    x,P = update(x,P, observations[:,1],t[1],parameters,q,R,ω)
    #x_results[1,:] = x  
    for i=2:N

       # println(i, "  ", x[1], "  ", x[2]," ", x[3])
       # println(i, "  ", P[1,1], "  ", P[2,2]," ", P[3,3])


        observation = observations[:,i]
        #println(observation[1], "  ", observation[2]," ", observation[3])
        ti = t[i]
        x_predict,P_predict   = predict(x,P,parameters,ti,dt,Q)
        x,P,l                 = update(x_predict,P_predict, observation,ti,parameters,q,R,ω)

        likelihood +=l
        #x_results[i,:] = x 
         
      
     end 

    #return x_results, likelihood
    return likelihood
    
    return 1.0


end 


















function update(x,P, observation,t,parameters,q,R,ω)

    println("UPDATE")
    H = H_function(parameters,t,q,ω)
    println(size(observation))
    println(size(H))
    println(size(x))
    y = observation .- H*x 
    #println("The innovation is")
    #println(y)
    #println("The P covariance is")
    #println(P[1,1], " ", P[2,2], " ", P[3,3])
    S = H*P*H' .+ R 
    K = P*H'*inv(S)
    xnew = x .+ K*y

   

    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = I - (K*H)
    KH = K*H
    #Pnew = I_KH * P * I_KH' .+ K * R * K'
    Pnew = I_KH * P

   l = get_likelihood(S,y)


    return xnew, Pnew,l
end 


function predict(x,P,parameters,t,dt,Q)

    
    F = F_function(parameters,dt)
    T = T_function(parameters,t,dt)

    xp = F*x .+ T 


    Pp = F*P*F' + Q
    
    return xp,Pp

end 



function get_likelihood(P::Matrix{NF},innovation::Vector{NF}) where {NF<:AbstractFloat}

    #println("Getting likelihood:  ", P[1,1],"  ", P[2,2],"  ", P[3,3])
    M = size(P)[1]
   
    x = P \ innovation
   
    
    #everything is diagonal --det(S) = 0
    return -NF(0.5) * (only(transpose(innovation) * x) +M*log(NF(2.0)*π))

end 




