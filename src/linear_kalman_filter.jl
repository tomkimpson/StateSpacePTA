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


#    println("Running with omega = ", ω)

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
    #l = log_likelihood(S,y)
    l = residuals(y)
    
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



function residuals(innovation::Vector{NF}) where {NF<:AbstractFloat}


return -log(abs.(innovation⋅innovation))

end 

