


struct ParameterPriors{NF<:AbstractFloat}

    f :: Distributions.Uniform{NF}
   

end 

"""
For now the guessed parameters are just the true parameters used to generate the synthetic data
"""
function set_priors(NF)

    f1 = Uniform(1,10)
   
    return ParameterPriors{NF}(f1) #convert to type NF 


end 

