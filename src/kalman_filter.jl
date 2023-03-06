


"""
Given some data recover the state and determine the likelihood
"""
function kalman_filter(observations::Matrix{NF},
                       PTA::Pulsars,
                       parameters::GuessedParameters
                      ) where {NF<:AbstractFloat}

   
    
    @unpack q,dt,t,f0 = PTA # PTA does have some parameters that ultimatley we might want to estimate . 

    @unpack σm,σp,γ,n,ω,Φ0 = parameters 


    #Set the dimension of the state space 
    L = size(observations)[1]     # dimension of hidden states i.e. number of pulsars
    N = size(observations)[2]     # number of timesteps

    #Get the 0th order frequencies and reshape them 
    #These are the PSR frequencies given by ANTF
    f0 = reshape(f0,(1,size(f0)[1])) #change f0 from a 1D vector to a 2D matrix

    #Initialise x and P
    x= observations[:,1] # guess that the intrinsic frequency is the same as the measured frequency
    #P = I(L) * σm*1e9 
    
    tmp_sigma = 1e-13
    P = diagm(fill(tmp_sigma^2*1e9 ,L)) 
    #Initialise the weights for thexite UKF
    ukf_weights = calculate_UKF_weights(NF,L,7e-4,2.0,0.0) #standard UKF parameters. Maybe make this arguments to be set?


    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,n,f0,dt,σp)


    # Calculate GW-related quantities that are time-constant
    GW = gw_variables(NF,parameters)
    prefactor,dot_product = gw_prefactor(GW.Ω,q,GW.Hij,ω,parameters.d)


    #Calculate measurement noise matrix
    R = R_function(L,σm)

    #Initialise an array to hold the results
    x_results = zeros(NF,N,L)
    println("LENGTH OF N = ", N)

    for i=1:N

        observation = observations[:,i]
        println("Observation number: ", i)
        observation = reshape(observation,(1,size(observation)[1])) #can we avoid these reshapes and get a consistent dimensionality?
        ti = t[i]

        
 
        
        # #Calculate sigma points, given the state variables 
        χ = calculate_sigma_vectors(x,P,ukf_weights.γ,L)
       


        # #Evolve the sigma points in time
        χ1 = F_function(χ,dt,parameters)




        
        # #Weighted state predictions and covariance

        x_predicted, P_xx,Δx= predict(χ1, ukf_weights)
        P_xx += Q


        #Evolve the sigma vectors according to the measurement function
        #Note we are omitting Joe's extra step recalculating sigmas here 
        χ2 = H_function(χ,ti,dot_product,prefactor,ω,Φ0)
        y_predicted, P_yy,Δy= predict(χ2, ukf_weights)
        P_yy += R
        
        #Measurement update 
        x,P = update(Δx, Δy,ukf_weights,P_xx,P_yy,observation,x_predicted,y_predicted)


        println("x: ", x[1])
        #Write to IO array 
        x_results[i,:] = x 
    
    end 
    return x_results

    

end 



function calculate_sigma_vectors(x::Vector{NF},P::Matrix{NF},γ::NF,L::Int) where {NF<:AbstractFloat}


  
   


  

    #eps_matrix = diagm(fill(eps() ,47)) 
    #Pstable = NF(0.5)*(P + transpose(P)) + eps_matrix

    
    #https://discourse.julialang.org/t/is-this-a-bug-with-cholesky/16970/2
    Psqrt = cholesky(Hermitian(P),check=true).L #the lower triangular part. This bit is returned by default in Python: https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
    M = γ .* Psqrt

  

    χ = zeros(NF,2*L+1,L)
    #χ = zeros(NF,L,2*L+1) #may want to go other way round for speed?

    χ[1,:] = x 
    χ[2:L+1,:] .= x' .+ M #to every row of M, add the vector x 
    χ[L+1+1:2*L+1,:] .= x' .- M 

    
    return χ
end 




"""
Sttruct to hold the UKF weights 
"""
struct UKF_weights{NF<:AbstractFloat}
    Wm :: Matrix{NF} 
    Wc :: Matrix{NF}
    γ :: NF
   
end 

"""
Calculate the weight vectors used in the UKF
"""
function calculate_UKF_weights(NF::Type, L::Int,α::Float64,β::Float64,κ::Float64)

    

    # Scaling parameter used in calculating the weights
    λ = α^2 *(L+κ) - L 
    q = 1/(2*(L+λ))

    #Weights Wm and Wc
    Wm = fill(q,(1,2*L+1)) 
    Wc = copy(Wm)

    Wm[1] = λ/(L + λ)
    Wc[1] = λ/(L + λ) + (NF(1.0)-α^2 + β)
    
    #And also define a γ. This is a different γ to the one used for pular state evolution!
    γ = sqrt(L + λ)

    return UKF_weights{NF}(Wm,Wc,γ)
   
end 



function predict(χ::Matrix{NF}, W::UKF_weights) where {NF<:AbstractFloat}


    @unpack Wm,Wc = W
    
    x_predicted = Wm * χ #this is a dot product 

   

    y = χ .- x_predicted #from each row of χ, subtract the x predictions


    yT = transpose(y)
    tmp = transpose(Wc) .* y #for each row in y, multiply it by the corresponding weight
    P_predicted = yT * tmp # size 47 x 47

    
    eps_matrix = diagm(fill(eps(NF) ,47)) 
    
    P_predicted = NF(0.5)*(P_predicted + transpose(P_predicted)) + eps_matrix # Enforce symmetry of the covariance matrix
   
    return x_predicted, P_predicted,y

end 




"""
Measurement update equations from Eric A. Wan and Rudolph van der Merwe
"""
function update(Δx::Matrix{NF}, Δy::Matrix{NF},W::UKF_weights, P_xx::Matrix{NF},P_yy::Matrix{NF},
               observation::Matrix{NF},x_predicted::Matrix{NF},y_predicted::Matrix{NF}) where {NF<:AbstractFloat}

    @unpack Wc =W

    #Get the cross correlation matrix
    yT = transpose(Δy)
    tmp = transpose(Wc) .* Δx 
    P_xy = yT * tmp # size 47 x 47
    #eps_matrix = diagm(fill(eps() ,47)) 

    #P_xy = NF(0.5)*(P_xy + transpose(P_xy)) + eps_matrix# Enforce symmetry of the covariance matrix

  



    #Get the Kalman gain. Do we need these transposes? 
    Q,R = qr(P_yy)
    Qb = transpose(Q)*P_xy
    


    K1 = transpose(R \ Qb) #left division operator solves R*K = Qb ; Ax = b
    K2 = P_yy \ P_xy
    K3 = transpose(K2)


    K = K1

   

    #Update state and covariance estiamtes
    
    innovation = transpose(observation - y_predicted)
    


    x = x_predicted + transpose(K*innovation)
    P = P_xx - K*P_yy*transpose(K)
   


    return vec(x),P
end 


