


struct GuessedParameters{NF<:AbstractFloat}

    #Pulsar parameters. Some of these are now matrices (i.e. 2D) to enable efficient operations in the Kalman filter
    d :: Vector{NF}
    γ :: Matrix{NF}
    n :: Matrix{NF} 

    #GW parameters 
    ω  ::NF     
    Φ0 ::NF      
    ψ  ::NF
    ι  ::NF
    δ  ::NF
    α  ::NF
    h  ::NF

    #Noise parameters 
    σp::NF
    σm::NF

end 

"""
For now the guessed parameters are just the true parameters used to generate the synthetic data
"""
function guess_parameters(pulsars::Pulsars,P::SystemParameters)

    @unpack ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,σp_guess,σm_guess = P
    @unpack γ,n,d = pulsars
    
    dims = (1,size(d)[1]) #1,Npulsar
   
    return GuessedParameters{P.NF}(d,reshape(γ,dims),reshape(n,dims),
                                   ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,σp_guess,σm_guess) #convert to type NF 


end 

