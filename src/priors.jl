


# struct ParameterPriors{NF<:AbstractFloat}

#     f :: Distributions.Uniform{NF}
   

# end 

# """
# For now the guessed parameters are just the true parameters used to generate the synthetic data
# """
# function set_priors(NF)

#     f1 = Uniform(1,10)
   
#     return ParameterPriors{NF}(f1) #convert to type NF 


# end 






# abstract type InferenceParameters end
# struct InferenceParameters{NF<:AbstractFloat} <: InferenceParameters
#     known_parameters::KnownParameters
#     unknown_parameters::Constants{NF}
# end



struct KnownParameters{NF<:AbstractFloat}

    #Pulsar parameters 
    #None of these are known

    #Gw parameters
    Φ0 ::NF      
    ψ  ::NF
    ι  ::NF
    δ  ::NF
    α  ::NF
    h  ::NF

    # #Noise parameters 
    σp::NF
    σm::NF

end 

"""
For now the guessed parameters are just the true parameters used to generate the synthetic data
"""
function set_known_parameters(P::SystemParameters)
    @unpack Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,σp_guess,σm_guess = P
    return KnownParameters{P.NF}(
                                   Φ0_guess,ψ_guess,ι_guess,δ_guess,α_guess,h_guess,
                                   σp_guess,σm_guess) #convert to type NF 

end 


# struct UnKnownParameters{NF<:AbstractFloat}





# end 



function set_pulsar_priors(PTA::Pulsars)

    @unpack f0, ḟ0, d, γ = PTA

    frac = 0.10
    f_prior = [set_distribution_limits(x,frac) for x in f0]
    ḟ_prior = [set_distribution_limits(x,frac) for x in ḟ0]
    d_prior = [set_distribution_limits(x,frac) for x in d]
    γ_prior = [set_distribution_limits(x,frac) for x in γ]
    return [f_prior;ḟ_prior;d_prior;γ_prior]
end 




function set_distribution_limits(x,frac)

        chunk = x*frac
        
        lower = x - chunk 
        upper = x + chunk

        if x < 0
            prior = Uniform(upper,lower)
        else
            prior = Uniform(lower,upper)
        end 
        

        return prior
end 




# function unpack_priors()
    
    

# end 







    # #Pulsar parameters. 
    # f0 :: Vector{NF} 
    # ḟ0 :: Vector{NF}
    # d :: Vector{NF}
    # γ :: Vector{NF}

    # #GW parameters 
    # ω  ::NF  






















