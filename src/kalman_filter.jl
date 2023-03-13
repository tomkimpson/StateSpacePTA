


"""
Given some data recover the state and determine the likelihood
"""
function kalman_filter(observations::Matrix{NF},
                       PTA::Pulsars,
                       parameters::GuessedParameters,
                       model::Symbol
                      ) where {NF<:AbstractFloat}

   
    @info "Running the Kalman filter for the measurement model defined via: ", model

    @unpack q,dt,t,f0 = PTA # PTA does have some parameters that ultimatley we might want to estimate . 

    @unpack σm,σp,γ,n,ω,Φ0 = parameters 



    #Set the dimension of the state space 
    Npulsars = size(observations)[1]
    L = Npulsars + 7 # dimension of hidden states i.e. number of pulsars + number of GW parameters 
    N = size(observations)[2]     # number of timesteps

    #Get the 0th order frequencies and reshape them 
    #These are the PSR frequencies given by ANTF
    f0 = reshape(f0,(1,size(f0)[1])) #change f0 from a 1D vector to a 2D matrix

    #Initialise x and P
    x_pulsar = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    x_parameters = [parameters.h, parameters.ι, parameters.δ, parameters.α, parameters.ψ, parameters.ω, parameters.Φ0] 

    println("The input x_parameters = ")
    println(x_parameters)
    x = [x_pulsar; x_parameters] #concatenate to get the intial state
    #P = I(L) * σm*1e9 
    
    tmp_sigma = NF(1e-3)
    P = diagm(fill(tmp_sigma ,L)) #maybe we want the uncertainty in the frequencies and the uncertainty in the parameters to be different?
    # #Initialise the weights for thexite UKF
    ukf_weights = calculate_UKF_weights(NF,L,7e-4,2.0,0.0) #standard UKF parameters. Maybe make this arguments to be set?


    #Calculate the time-independent Q-matrix
    Q = Q_function(γ,n,f0,dt,σp,7)


    # # Calculate GW-related quantities that are time-constant
    # GW = gw_variables(NF,parameters)
    # prefactor,dot_product = gw_prefactor(GW.Ω,q,GW.Hij,ω,parameters.d)


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

    

     for i=1:N

        println("STEP NUMBER i ", i)
         #Grab the observatin and the time 
         observation = observations[:,i]
         observation = reshape(observation,(1,size(observation)[1])) #can we avoid these reshapes and get a consistent dimensionality?
         ti = t[i]

         # #Calculate sigma points, given the state variables
         χ = calculate_sigma_vectors(x,P,ukf_weights.γ,L)
        
       
         # #Evolve the sigma points in time
         χ_input = χ[:,1:Npulsars]              #Get the columns which correspond to the frequencies 
         χ_remain = χ[:,Npulsars+1:Npulsars+7]  #Save the columns which correspond to the parameters 
        
         
         χ1 = F_function(χ_input,dt,parameters)
        

         χ1 = [χ1 χ_remain]


  

        #  #Weighted state predictions and covariance
         x_predicted, P_xx,Δx= predict(χ1, ukf_weights)
         P_xx += Q

         println("x predicted")
         println(size(x_predicted), " ", size(P_xx))

         #Evolve the sigma vectors according to the measurement function
         #Note we are omitting Joe's extra step recalculating sigmas here.
         #Including this steps makes the performance worse -
         # after discussion we think the two methods are equivalent 
         #χ2 = measurement_function(χ,ti,dot_product,prefactor,ω,Φ0)
         χ2 = measurement_function(χ,ti,Npulsars,PTA.q,PTA.d)
 
         y_predicted, P_yy,Δy= predict(χ2, ukf_weights)
         println("y_predicted")
         println(size(y_predicted), " ", size(P_yy))
         

         P_yy += R
        
        #  #Measurement update 
        println("update")
         x,P,l = update(Δx, Δy,ukf_weights,P_xx,P_yy,observation,x_predicted,y_predicted)

         println(x)
         #Do some IO, update likelihood 
         x_results[i,:] = x 
         likelihood += l
      
     end 

    return x_results, likelihood
    #return 1,1

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

    #println(χ[1,:])
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
    P_predicted = yT * tmp # size L x L

    
    eps_matrix = diagm(fill(eps(NF) ,size(x_predicted)[2]))  #size(x_predicted)[2] = L 
    
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
    

    #K1 = transpose(R \ Qb) #left division operator solves R*K = Qb ; Ax = b
    K2 = P_yy \ P_xy
    K3 = transpose(K2)


    K = K3


   
    #Update state and covariance estiamtes    
    innovation = transpose(observation - y_predicted)
    


    x = x_predicted + transpose(K*innovation)
    P = P_xx - K*P_yy*transpose(K)
   
    l = likelihood(P_yy,innovation)


    return vec(x),P,l
end 


function likelihood(P_yy::Matrix{NF},innovation::LinearAlgebra.Transpose{NF, Matrix{NF}}) where {NF<:AbstractFloat}

    M = size(P_yy)[1] #number of observations per timestep i.e. number of pulsars
    x = P_yy \ innovation
    return -NF(0.5) * (logdet(P_yy) + only(transpose(innovation) * x) +M*log(NF(2.0)*π))

end 