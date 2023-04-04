


# function set_priors(θ::GuessedParameters)



#     @unpack f0,ḟ0,d,γ,ω,Φ0,ψ,ι,δ,α,h,σp,σm = θ 

#     # Set the priors on each of the pulsar parameters
#     f_prior = Dirac.(f0)
#     ḟ_prior = Dirac.(ḟ0)
#     d_prior = Dirac.(d)
#     γ_prior = Dirac.(γ)

#     # GW parameters 
#     ω_prior  = LogUniform(1e-9,1e-6)
#     Φ0_prior = Dirac(Φ0)  
#     ψ_prior  = Dirac(ψ)
#     ι_prior  = Dirac(ι)
#     δ_prior  = Dirac(δ)
#     α_prior  = Dirac(α)
#     h_prior  = Dirac(h)
    
#     # Noise parameters 
#     σp_prior= Dirac(σp)
#     σm_prior= Dirac(σm)


#     #Concat. There is probably a clever way to do this...
#     priors = [f_prior ; ḟ_prior; d_prior;γ_prior; ω_prior;Φ0_prior;ψ_prior;ι_prior;δ_prior;α_prior;h_prior;σp_prior;σm_prior]
 
#     f_names = name_those_parameters(f_prior,"f")
#     ḟ_names = name_those_parameters(f_prior,"ḟ")
#     d_names = name_those_parameters(f_prior,"d")
#     γ_names = name_those_parameters(f_prior,"γ")
 
#     prior_names = [f_names ; ḟ_names; d_names; γ_names; "ω";"Φ0";"ψ";"ι";"δ";"α";"h";"σp";"σm"]



#     @assert length(priors) == length(prior_names)
#     return priors,prior_names
   
#     # return GuessedParameters{P.NF}(f0,ḟ0,d,γ,
#     #                                ω ,Φ0 ,ψ ,ι ,δ ,α ,h ,
#     #                                σp ,σm ) #convert to type NF 


# end 













# function name_those_parameters(x,label)    
#     names = [label*string(i) for i=1:length(x)]
#     return names
# end 









# end 

# function set_pulsar_priors(PTA::Pulsars)

#     @unpack f0, ḟ0, d, γ = PTA

#     frac = 0.001
#     f_prior = [set_distribution_limits(x,frac) for x in f0]
#     ḟ_prior = [set_distribution_limits(x,frac) for x in ḟ0]
#     d_prior = [set_distribution_limits(x,frac) for x in d]
#     γ_prior = [set_distribution_limits(x,frac) for x in γ]
#     return [f_prior;ḟ_prior;d_prior;γ_prior]
# end 


# function set_single_pulsar_priors(PTA::Pulsars)

#     @unpack f0, ḟ0, d, γ = PTA

#     return [f0;ḟ0; d; γ]
# end 



# function set_distribution_limits(x,frac)

#         chunk = x*frac
        
#         lower = x - chunk 
#         upper = x + chunk

#         if x < 0
#             prior = Uniform(upper,lower)
#         else
#             prior = Uniform(lower,upper)
#         end 
        

#         return prior
# end 




# function unpack_priors()
    
    

# end 







    # #Pulsar parameters. 
    # f0 :: Vector{NF} 
    # ḟ0 :: Vector{NF}
    # d :: Vector{NF}
    # γ :: Vector{NF}

    # #GW parameters 
    # ω  ::NF  






















