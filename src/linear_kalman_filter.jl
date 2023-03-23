"""
Given some observations and a specification of the PTA and parameters, 
recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
             PTA::Pulsars,
             parameters::GuessedParameters,
             model::Symbol
             ) where {NF<:AbstractFloat}

 
    @unpack q,dt,t = PTA                                  # Get the known parameters related to the PTA    
    @unpack ω, h,ι,δ,α,ψ,Φ0,σp,σm,f0,ḟ0,d,γ = parameters  # All unknown parameters

    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    N = size(observations)[2]     # number of timesteps
  
    #Initialise x and P
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = I(L) * 1e-6*1e9  #P = I(L) * σm*1e9 
   
    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,σp,dt)
    
    #Calculate measurement noise matrix
    R = R_function(Npulsars,σm)

    #Initialise an array to hold the results
    x_results = zeros(NF,N,L)

    #Initialise a likelihood variable
    likelihood = NF(0.0)

    #Define the time-constant GW quantities
    m,n,n̄,Hij = gw_variables(h,ι, δ, α, ψ)
    prefactor,dot_product = gw_prefactor(n̄,q,Hij,ω,d)


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
    x,P = update(x,P, observations[:,1],t[1],R,ω,Φ0,prefactor,dot_product,measurement_function)



    
    x_results[1,:] = x 
    for i=2:N
        observation           = observations[:,i] #column-major slicing
        ti                    = t[i]
        x_predict,P_predict   = predict(x,P,f0,ḟ0,γ,ti,dt,Q)
        x,P,l                 = update(x_predict,P_predict, observation,ti,R,ω,Φ0,prefactor,dot_product,measurement_function)

        likelihood +=l
        x_results[i,:] = x 
         
         
     end 

    
    return likelihood,x_results
    

end 



"""
Given some observations and a specification of the PTA and parameters, 
recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
             PTA::Pulsars,
             parameters::Vector{NF},
             model::Symbol
             ) where {NF<:AbstractFloat}

 
    @unpack q,dt,t = PTA     
          
    ω, h,ι,δ,α,ψ,Φ0,σp,σm,f0,ḟ0,d,γ = read_vector(parameters)  # All unknown parameters

    #println(ω, " ", h, " ", " ", ι," ",δ, " ",α, " ", ψ," ",Φ0," ",σp, " ",σm, " ", f0, " ",ḟ0, " ",d," ",γ)
    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars 
    N = size(observations)[2]     # number of timesteps
  
    #Initialise x and P
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = I(L) * 1e-6*1e9  #P = I(L) * σm*1e9 
   
    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,σp,dt)
    
    #Calculate measurement noise matrix
    R = R_function(Npulsars,σm)

    #Initialise an array to hold the results
    x_results = zeros(NF,N,L)

    #Initialise a likelihood variable
    likelihood = NF(0.0)

    #Define the time-constant GW quantities
    m,n,n̄,Hij = gw_variables(h,ι, δ, α, ψ)
    prefactor,dot_product = gw_prefactor(n̄,q,Hij,ω,d)


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
    x,P = update(x,P, observations[:,1],t[1],R,ω,Φ0,prefactor,dot_product,measurement_function)



    
    x_results[1,:] = x 
    for i=2:N
        observation           = observations[:,i] #column-major slicing
        ti                    = t[i]
        x_predict,P_predict   = predict(x,P,f0,ḟ0,γ,ti,dt,Q)
        x,P,l                 = update(x_predict,P_predict, observation,ti,R,ω,Φ0,prefactor,dot_product,measurement_function)

        likelihood +=l
        x_results[i,:] = x 
         
         
     end 

    
    return likelihood,x_results
    

end 












function update(x::Vector{NF},
                P::LinearAlgebra.Diagonal{NF, Vector{NF}}, 
                observation::Vector{NF},
                t::NF,
                R::LinearAlgebra.Diagonal{NF, Vector{NF}},
                ω::NF,
                Φ0::NF,
                prefactor::Vector{NF},
                dot_product::Vector{NF},
                measurement_function::Function) where {NF<:AbstractFloat}

    #Define the measurement matrix
    H = measurement_function(t, ω,Φ0,prefactor,dot_product)

    #And its transpose 
    # we can set this as H', but our H is diagonal. Is Julia smart enough to make the optimisation itself, or should we set manually? H vs H' vs transpose(H)
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
    # ...despite the extra cost of operations
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
    return -NF(0.5) * (logdet(P) + only(transpose(innovation) * x) +M*log(NF(2.0)*π))
end 


function read_vector(x::Vector{NF}) where {NF<:AbstractFloat}

    # Vector of length x 
    # The last 9 elements are single parameters shared between every pulsar, including σp 
    # The first elements are f0,ḟ0,d,γ
    Npsr = Int64((length(x) - 9) / 4)


    f0 = x[1:Npsr]
    ḟ0 = x[Npsr+1:2*Npsr]
    d  = x[2*Npsr+1:3*Npsr]
    γ  = x[3*Npsr+1:4*Npsr]

    ω  = x[4*Npsr + 1]
    Φ0 = x[4*Npsr + 2]
    ψ  = x[4*Npsr + 3]
    ι  = x[4*Npsr + 4]
    δ  = x[4*Npsr + 5]
    α  = x[4*Npsr + 6]
    h  = x[4*Npsr + 7]
    σp = x[4*Npsr + 8]
    σm = x[4*Npsr + 9]



    return ω, h,ι,δ,α,ψ,Φ0,σp,σm,f0,ḟ0,d,γ 


end 

