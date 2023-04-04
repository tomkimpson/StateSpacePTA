"""
Given some observations and a specification of the PTA and parameters, 
recover the state and determine the likelihood
"""
function KF(observations::Matrix{NF},
            PTA::Pulsars,
            parameters::KalmanParameters
            ) where {NF<:AbstractFloat}

 

    @unpack q,d,dt,t = PTA 
    @unpack γ, σp,σm,f0,ḟ0,δ,α,ψ,h,cos_ι,ω,Φ0 = parameters 

    #Precompute all the transition and control matrices as well as Q and R matrices.
    #F,Q,R are time-independent functions of the parameters
    #T is time dependent, but does not depend on states and so can be precomputed    
    Q = Q_function(γ,σp,dt)
    R = R_function(σm)
    F = F_function(γ,dt)
    T = T_function(f0,ḟ0,γ,t,dt) #ntimes x npulsars



    #Initialise x and P 
    x = observations[:,1] # guess that the intrinsic frequencies is the same as the measured frequency
    P = ones(length(x)) * σm*1e10

    #Precompute the influence of the GW
    #This does not depend on the states, only the parameters 
    gw_factor = gw_frequency_modulation_factor(δ,α,ψ,h,cos_ι,ω,Φ0,q,d,t)

    #Initialise the likelihood
    likelihood = NF(0.0)

    #Initialise an array to hold the results
    x_results = zero(observations)  #could also use similar?

    
    #Perform the first update step 
    i=1
    x,P,l = update(x,P, observations[:,i],R,gw_factor[:,i])
    likelihood += l
    x_results[:,i] = x 
    
    for i =2:length(t)
        x_predict, P_predict = predict(x,P,F,T[:,i],Q)
        x,P,l                = update(x_predict,P_predict, observations[:,i],R,gw_factor[:,i]) 
        likelihood += l
        x_results[:,i] = x 
    end


    return likelihood,x_results
end 


function update(x::Vector{NF},P::Vector{NF},observation::Vector{NF},R::NF,H::Vector{NF}) where {NF<:AbstractFloat}


    y    = observation .- H.*x #innovatin
    S    = H.*P.*H .+ R        #innovatin covariance 
    K    = P.*H ./S            #Kalman gain
    xnew = x .+ K.*y            #updated state
    I_KH = NF(1.0) .- K.*H
    Pnew = I_KH .* P .* I_KH .+ K .* R .* K #updated covariance

    #...and get the likelihood
    l = log_likelihood(S,y)
    
    return xnew,Pnew,l

end 

function predict(x::Vector{NF},
                 P::Vector{NF},
                 F::Vector{NF},
                 T::Vector{NF},
                 Q::Vector{NF} 
               ) where {NF<:AbstractFloat}

    xp = F.*x .+ T 
    Pp = F.*P.*F .+ Q
    return xp,Pp

end 

function log_likelihood(S::Vector{NF}, innovation::Vector{NF}) where {NF<:AbstractFloat}
    N = length(S)
    x = innovation ./ S 
    slogdet = sum(log.(S))
    return -NF(0.5) * (slogdet + innovation⋅x) +N*log(NF(2.0)*π)
end 


# function read_vector(x::Vector{NF}) where {NF<:AbstractFloat}

#     # Vector of length x 
#     # The last 9 elements are single parameters shared between every pulsar, including σp 
#     # The first elements are f0,ḟ0,d,γ
#     Npsr = Int64((length(x) - 9) / 4)


#     f0 = x[1:Npsr]
#     ḟ0 = x[Npsr+1:2*Npsr]
#     d  = x[2*Npsr+1:3*Npsr]
#     γ  = x[3*Npsr+1:4*Npsr]

#     ω  = x[4*Npsr + 1]
#     Φ0 = x[4*Npsr + 2]
#     ψ  = x[4*Npsr + 3]
#     ι  = x[4*Npsr + 4]
#     δ  = x[4*Npsr + 5]
#     α  = x[4*Npsr + 6]
#     h  = x[4*Npsr + 7]
#     σp = x[4*Npsr + 8]
#     σm = x[4*Npsr + 9]



#     #return ω, h,ι,δ,α,ψ,Φ0,σp,σm,f0,ḟ0,d,γ 
#     return h,ι,δ,α,ψ,Φ0,σp,σm,f0,ḟ0,d,γ 


# end 

