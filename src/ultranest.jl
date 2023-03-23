




function parameter_estimation_ultranest(::Type{NF}=Float64;              # number format, use Float64 as default
                              kwargs...                        # all additional non-default parameters
                              ) where {NF<:AbstractFloat}


    println("hello from parameter_estimation()")

    state,measurements,PTA,θ̂,P = setup(NF=NF;kwargs...)
    priors,prior_names         = set_priors(θ̂)
   

    likelihood = let observations = measurements, PTA = PTA

        params -> begin 
            ll_value,x_results = KF(observations,
                                    PTA,
                                    params,
                                    :GW)
            return ll_value
        end
    end

   
    
    model = NestedModel(likelihood, priors)
    spl = Nested(length(priors), 50) #X parameters, Y active points, multi-ellipsoid. See docstring
    display(prior_names)

    chain, state = sample(model, spl; dlogz=0.2, param_names=prior_names)

    # optionally resample the chain using the weights
    chain_res = sample(chain, Weights(vec(chain["weights"])), length(chain))

    return chain,state,chain_res


end 