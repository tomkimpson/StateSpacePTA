


"""
Given some data recover the state and determine the likelihood
"""
function kalman_filter(observations::Matrix{NF},
                       PTA::Pulsars,
                       parameters::GuessedParameters
                      ) where {NF<:AbstractFloat}

    i = 1
    
    @unpack q,dt,t = PTA #PTA does have some parameters that we might want to estimate ultimatley. Unpack it all here
    
    #Set the dimension of the state space 
    L = size(observations)[1]     # dimension of hidden states i.e. number of pulsars

    ti = t[i]

    #Initialise x and P
    x= observations[:,i] # guess that the intrinsic frequency is the same as the measured frequency
    P = I(L) * parameters.σm*1e9 
   

    #Initialise the weights for the UKF
    ukf_weights = calculate_UKF_weights(NF,L,7e-4,2.0,0.0) #standard UKF parameters. Maybe make this arguments to be set?


    χ = calculate_sigma_vectors(x,P,ukf_weights.γ,L)

    F_function(χ,dt,parameters)

    GW = gw_variables(NF,parameters)
    prefactor,dot_product = gw_prefactor(GW.Ω,
                                         q,GW.Hij,parameters.ω,parameters.d)

    println("dot product")
    println(size(dot_product))
    println(size(prefactor))
    out = H_function(χ,ti,dot_product,prefactor,parameters.ω,parameters.Φ0)
    println(size(out))
    #for i=1:size(observations)[2]

     #   obs = observations[:,i] #a vector of observations over N pulsars at time t 

    #println(i)

    #end
   #for obs in observations
   # println(size(obs))



  # end 

  println("-------END OF KALMAN FILTER---------------")
end 



function calculate_sigma_vectors(x::Vector{NF},P::LinearAlgebra.Diagonal{NF, Vector{NF}},γ::NF,L::Int) where {NF<:AbstractFloat}

    Psqrt = cholesky(P,check=true).L #the lower triangular part. This bit is returned by default in Python: https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
    M = γ .* Psqrt

    χ = zeros(NF,2*L+1,L)

    χ[1,:] = x 
    χ[2:L+1,:] .= x .+ M
    χ[L+1+1:2*L+1,:] .= x .- M

   
    return χ
end 







struct UKF_weights{NF<:AbstractFloat}

    Wm :: Vector{NF} 
    Wc :: Vector{NF}
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
    Wm = fill(q,2*L+1) 
    Wc = copy(Wm)

    Wm[1] = λ/(L + λ)
    Wc[1] = λ/(L + λ) + (1.0-α^2 + β)
    
    #And also define a γ. This is a different γ to the one used for pular state evolution!
    γ = sqrt(L + λ)

    return UKF_weights{NF}(Wm,Wc,γ)
   
end 

# def _calculate_weights(self):

# """
# Internal function

# Calculate the weights of the UKF.

# Updates self.Wm, self.Wc
# """

  
# lambda_ = self.alpha**2 *(self.L+self.kappa) - self.L # scaling parameter used in calculating the weights

# #Preallocating arrays then filling to make dimensions explicit.
# #Verbose, but clear. Maybe just use np.full()...
# self.Wm = np.zeros(2*self.L+1,dtype=NF)  
# self.Wc = np.zeros(2*self.L+1,dtype=NF)



# #Fill Wm
# self.Wm[0] = lambda_  / (self.L + lambda_ )
# for i in range(1,len(self.Wm)):
#     self.Wm[i] = 1/(2*(self.L+lambda_ ))

# #Fill Wc
# self.Wc[0] = lambda_  / (self.L + lambda_ ) + (1 - self.alpha**2 + self.beta)
# for i in range(1,len(self.Wc)):
#     self.Wc[i] = 1/(2*(self.L+lambda_ ))




# #Also define....
# self.gamma = np.sqrt(self.L + lambda_,dtype=NF)


















# """
# Internal function

# Calculate the sigma vectors for a given state `x` and covariance `P`

# Updates self.chi

# See Eq. 15 from Wav/Van
# """

# P_check = P

# P_sqrt = la.cholesky(P_check, check_finite=True)  # cholesky is much faster than scipy.linalg.sqrtm. Note that this is float64 not float128





# M = self.gamma*P_sqrt

# self.chi = np.zeros((2*self.L + 1,self.L),dtype=NF)


# self.chi[0,:] = x

# self.chi[1:self.L+1,:] = x + M
# self.chi[self.L+1:2*self.L+1+1,:] = x - 


# end 