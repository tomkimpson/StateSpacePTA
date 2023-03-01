


"""
Given some data recover the state and determine the likelihood
"""
function kalman_filter(observations::Matrix{NF},σm::NF) where {NF<:AbstractFloat}


    

    L = size(observations)[1]     # dimension of hidden states i.e. number of pulsars


    i = 1

    #Initialise x and P
    x= observations[:,i] # guess that the intrinsic frequency is the same as the measured frequency
    println(typeof(x))
    P = I(L) * σm*1e9 # a square matrix, dim(L x L). # How to initialise?
    println(size(P))
   
    println(typeof(P))

    calculate_sigma_vectors(x,P)
    #println(P)
    #Initialise state and covariance 



    #for i=1:size(observations)[2]

     #   obs = observations[:,i] #a vector of observations over N pulsars at time t 

    #println(i)

    #end
   #for obs in observations
   # println(size(obs))



  # end 

end 



function calculate_sigma_vectors(x::Vector{NF},P::LinearAlgebra.Diagonal{NF, Vector{NF}}) where {NF<:AbstractFloat}


println("this is calcu")

C = cholesky(P)


println(typeof(C))
end 

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