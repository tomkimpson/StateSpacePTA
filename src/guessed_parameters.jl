


struct KalmanParameters{NF<:AbstractFloat}

    #Pulsar parameters. 
    f0 :: Vector{NF} 
    ḟ0 :: Vector{NF}
    d :: Vector{NF}
    γ :: Vector{NF}

    #GW parameters 
    ω  ::NF     
    Φ0 ::NF      
    ψ  ::NF
    cos_ι  ::NF
    δ  ::NF
    α  ::NF
    h  ::NF

    # #Noise parameters 
    σp::NF
    σm::NF

end 

"""
Set the parameters that will be passed to the Kalman filter For now the guessed parameters are just the true parameters used to generate the synthetic data
"""
function kalman_parameters(pulsars::Pulsars,P::SystemParameters)

    @unpack ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,σp_guess,σm_guess = P
    @unpack f0,ḟ0,d,γ = pulsars
    
   
    return KalmanParameters{P.NF}(f0,ḟ0,d,γ,
                                  ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,
                                  σp_guess,σm_guess) #convert to type NF 


end 


# function guess_parameters(f::Vector{Float64}, pulsars::Pulsars,P::SystemParameters)

#     @unpack ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,σp_guess,σm_guess = P
#     @unpack ḟ0,d,γ = pulsars

#     f0 = f
    
   
#     return GuessedParameters{P.NF}(f0,ḟ0,d,γ,
#                                    ω_guess,Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,
#                                    σp_guess,σm_guess) #convert to type NF 


# end 


