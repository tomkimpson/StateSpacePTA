


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

    for i=1:3 #40 #size(observations)[2]

        observation = observations[:,i]
        println("Observation number: ", i)
        #println("Observations for this timestep are:")
        #display(observation)
        observation = reshape(observation,(1,size(observation)[1])) #can we avoid these reshapes and get a consistent dimensionality?
        ti = t[i]

        
 
        
        # #Calculate sigma points, given the state variables 
        χ = calculate_sigma_vectors(x,P,ukf_weights.γ,L)
       
        # println("Sigma vector second row")
        # println(size(χ))
        # display(χ[2,:])

        # #Evolve the sigma points in time
        χ1 = F_function(χ,dt,parameters)


        # println("Evolved sigma1 vector second row ", dt)
        # println(size(χ1))
        # display(χ1[2,:])

        
        # #Weighted state predictions and covariance
        println("---------GET X PREDICT")
        x_predicted, P_xx,Δx= predict(χ1, ukf_weights)
        P_xx += Q


        # println("P_xx is")
        # display(P_xx)
        # print("_--------------------------")
        #Evolve the sigma vectors according to the measurement function
        #Note we are omitting Joe's extra step recalculating sigmas here 
        χ2 = H_function(χ,ti,dot_product,prefactor,ω,Φ0)
        println("---------GET Y PREDICT")
        y_predicted, P_yy,Δy= predict(χ2, ukf_weights)
        P_yy += R
        
        #Measurement update 
        x,P = update(Δx, Δy,ukf_weights,P_xx,P_yy,observation,x_predicted,y_predicted)


        println("The output x is = ",)
        println(x)
        println("The output of the 2nd row of P is =")
        println(P[2,:])

        println("============END OF ITERATION========================")

    end 

    

  println("-------END OF KALMAN FILTER---------------")
end 



function calculate_sigma_vectors(x::Vector{NF},P::Matrix{NF},γ::NF,L::Int) where {NF<:AbstractFloat}


  
    println("Calculate the sigma vectors")


  

    #eps_matrix = diagm(fill(eps() ,47)) 
    #Pstable = NF(0.5)*(P + transpose(P)) + eps_matrix

    #println("Attempting Cholesky of P matrix")
    #display(diag(P))
    #display(P)
    #display(diag(P))
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
    Wc[1] = λ/(L + λ) + (1.0-α^2 + β)
    
    #And also define a γ. This is a different γ to the one used for pular state evolution!
    γ = sqrt(L + λ)

    return UKF_weights{NF}(Wm,Wc,γ)
   
end 



function predict(χ::Matrix{NF}, W::UKF_weights) where {NF<:AbstractFloat}


    @unpack Wm,Wc = W
    
    x_predicted = Wm * χ #this is a dot product 

    # println("You are inside the predict function")
    # println("The weights used in the predict function are:")
    # println(Wm)
    # println("The sigma vector row used in the predict function is:")
    # println(χ[2,:])

    # println(χ[2,:])
    # println(Wm)
    # println(x_predicted)
    # println("**************************")
    
    #println(x_predicted)

    y = χ .- x_predicted #from each row of χ, subtract the x predictions


    yT = transpose(y)
    tmp = transpose(Wc) .* y #for each row in y, multiply it by the corresponding weight
    P_predicted = yT * tmp # size 47 x 47

    #println("the second row of the covar matrix is")
    
    #println("before_correction")
    #println(P_predicted[2,2], " ",P_predicted[2,3]," ",P_predicted[2,4])
    eps_matrix = diagm(fill(eps(NF) ,47)) 
    
    P_predicted = NF(0.5)*(P_predicted + transpose(P_predicted)) + eps_matrix # Enforce symmetry of the covariance matrix
    #println("after correction")
    #println(P_predicted[2,2]," ", P_predicted[2,3]," ",P_predicted[2,4])


    #println("-------------------------")

    println("The output from the predict function is")
    println(x_predicted)

    return x_predicted, P_predicted,y

end 




"""
Measurement update equations from Eric A. Wan and Rudolph van der Merwe
"""
function update(Δx::Matrix{NF}, Δy::Matrix{NF},W::UKF_weights, P_xx::Matrix{NF},P_yy::Matrix{NF},
               observation::Matrix{NF},x_predicted::Matrix{NF},y_predicted::Matrix{NF}) where {NF<:AbstractFloat}

    @unpack Wc =W
    println("You are inside the update function")

    #Get the cross correlation matrix
    yT = transpose(Δy)
    tmp = transpose(Wc) .* Δx 
    P_xy = yT * tmp # size 47 x 47
    #eps_matrix = diagm(fill(eps() ,47)) 

    #P_xy = NF(0.5)*(P_xy + transpose(P_xy)) + eps_matrix# Enforce symmetry of the covariance matrix

    #println("Is Pxy posdef?", " ", isposdef(P_xy))



    #Get the Kalman gain. Do we need these transposes? 
    Q,R = qr(P_yy)
    Qb = transpose(Q)*P_xy
    

    K1 = transpose(R \ Qb) #left division operator solves R*K = Qb ; Ax = b
    K2 = P_yy \ P_xy
    K3 = transpose(K2)


    K = K3
    #println("writing")
    #writedlm("KalmanGain.txt", K)
   

    #Update state and covariance estiamtes
    
    innovation = transpose(observation - y_predicted)
    

    println("x predicted is:")
    println(x_predicted)
    println("innovation is")
    println(innovation)
    println("Second line of Kalman gain is")
    println(K[2,:])
    x = x_predicted - transpose(K*innovation)
    P = P_xx - K*P_yy*transpose(K)
    blob_factor = K*P_yy*transpose(K)
    #writedlm("blob_factor.txt", blob_factor)

    #println("The first row of the new covar matrix is:")
    #println(P[1,:])


    return vec(x),P
end 


