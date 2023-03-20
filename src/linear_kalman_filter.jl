


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
    # if model == :GW
    #     measurement_function = H_function
    # elseif model == :null
    #     measurement_function = null_function
    # else
    #     println("Model is not defined. Choose one of :GW or :null" )
    #     return
    # end 
    
    #Update step first, then iterate over remainders
    #See e.g. https://github.com/meyers-academic/baboo/blob/f5619df23b2465373443e02cd52e1003ed66c0ac/baboo/kalman.py#L167
    #println(1, "  ", x[1], "  ", x[2]," ", x[3])
    #println(1, "  ", P[1,1], "  ", P[2,2]," ", P[3,3])
    
   # display(P)
    x,P = update(x,P, observations[:,1],t[1],parameters,q,R)

    #x_results[1,:] = x  
    for i=2:N
        #display(P)
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



# """
# Given some data recover the state and determine the likelihood
# """
# function KF(observations::Matrix{NF},
#              PTA::Pulsars,
#              parameters::GuessedParameters,
#              ω_tuple::Vector{Float64},
#              model::Symbol
#              ) where {NF<:AbstractFloat}
function KF(observations::Matrix{NF},
             PTA::Pulsars,
             known_parameters::KnownParameters,  # this is a struct with values set specifically. It can be unpacked using @unpack
             unknown_parameters::Vector{Float64} # this is a vector with values sampled from a distribution
             ) where {NF<:AbstractFloat}


    #println("WELCOME TO THE KALMAN FILTER")
    #println(typeof(unknown_parameters))
 
    @unpack q,dt,t = PTA         #Get the parameters related to the PTA 
    @unpack h,ι,δ,α,ψ,Φ0,σp,σm = known_parameters 

    print(unknown_parameters)
    #Unpack the unknow parameters 
    #Todo: make this a mapping function 
    ω = unknown_parameters[1]
    f0 = unknown_parameters[2:11]
    ḟ0 = unknown_parameters[12:21]
    d  = unknown_parameters[22:31]
    γ  = unknown_parameters[32:41]

    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    N = size(observations)[2]     # number of timesteps
  
    #Initialise x and P
    x = 1.12*observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = I(L) * 1e-6*1e9  #P = I(L) * σm*1e9 
   
    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,σp,dt)
    

    println("The q matrix is")
    #display(Q)
    #Calculate measurement noise matrix
    R = R_function(Npulsars,σm)

    #Initialise an array to hold the results
    x_results = zeros(NF,N,L)

    #Initialise a likelihood variable
    likelihood = NF(0.0)

    #Define the time-constant GW quantities
    m,n,n̄,Hij = gw_variables(h,ι, δ, α, ψ)
    prefactor,dot_product = gw_prefactor(n̄,q,Hij,ω,d)

    # #Set what measurement model to use
    # if model == :GW
    #      = H_function
    # elseif model == :null
    #     measurement_function = null_function
    # else
    #     println("Model is not defined. Choose one of :GW or :null" )
    #     return
    # end 
    
    #Update step first, then iterate over remainders
    #See e.g. https://github.com/meyers-academic/baboo/blob/f5619df23b2465373443e02cd52e1003ed66c0ac/baboo/kalman.py#L167


   
    x,P = update(x,P, observations[:,1],t[1],R,ω,Φ0,prefactor,dot_product)

    x_results[1,:] = x 
    for i=2:N
 

  

        observation           = observations[:,i] #column-major slicing
        ti                    = t[i]
        x_predict,P_predict   = predict(x,P,f0,ḟ0,γ,ti,dt,Q)
        x,P,l                 = update(x_predict,P_predict, observation,ti,R,ω,Φ0,prefactor,dot_product)

        likelihood +=l
        x_results[i,:] = x 
         
         
     end 


    return likelihood,x_results
    

end 


















function update(x::Vector{NF},
                P::LinearAlgebra.Diagonal{NF, Vector{NF}}, 
                observation::Vector{NF},
                t::NF,
                #parameters::GuessedParameters,
                #q::Matrix{Float64},
                R::LinearAlgebra.Diagonal{NF, Vector{NF}},
                ω::NF,
                Φ0::NF,
                prefactor::Vector{NF},
                dot_product::Vector{NF}) where {NF<:AbstractFloat}

    #Define the measurement matrix
    H = H_function(t, ω,Φ0,prefactor,dot_product)

    #And its transpose 
    ## we can set this as H', but our H is diagonal. Is Julia smart enough to make the optimisation itself, or should we set manually? H vs H' vs transpose(H)
    HT = H 

   
    y = observation .- H*x   # Innovation
    S = H*P*HT .+ R          # Innovation covariance 
    K = P*HT*inv(S)          # Kalman gain
    xnew = x .+ K*y          # Updated x

    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    # In practice for this pulsar problem I have also found this expression more numerically stable
    # ...despite the extra cost
    I_KH = I - (K*H)
    Pnew = I_KH * P * I_KH' .+ K * R * K'
   
    #And finally get the likelihood
    l = get_likelihood(S,y)
    return xnew, Pnew,l
end 


function predict(x::Vector{NF},
                 P::LinearAlgebra.Diagonal{NF, Vector{NF}},
                 #parameters::GuessedParameters,
                 f0::Vector{NF},
                 ḟ0::Vector{NF},
                 γ::Vector{NF},
                 t::NF,
                 dt::NF,
                 Q::LinearAlgebra.Diagonal{NF, Vector{NF}}) where {NF<:AbstractFloat}

    
    F = F_function(γ,dt)
    T = T_function(f0, ḟ0, γ,t,dt)

    xp = F*x .+ T 
    Pp = F*P*F' + Q
    return xp,Pp

end 



function get_likelihood(P::LinearAlgebra.Diagonal{NF, Vector{NF}},innovation::Vector{NF}) where {NF<:AbstractFloat}
    M = size(P)[1]
    x = P \ innovation
    #println("DET:", det(P))
    #everything is diagonal --logdet(S) = 0 ?
    return -NF(0.5) * (logdet(P) + only(transpose(innovation) * x) +M*log(NF(2.0)*π))

end 




