




function infer_parameters2(::Type{NF}=Float64;              # number format, use Float64 as default
        kwargs...                        # all additional non-default parameters
        ) where {NF<:AbstractFloat}



        P = SystemParameters(NF=NF;kwargs...) # Parameters
        PTA = setup_PTA(P)
        GW = gw_variables(P.NF,P)
    
        @info "Hello from StateSpacePTA. You are running with NF = ", P.NF
    
        seed = P.seed # Integer or nothing 
        state,measurement = create_synthetic_data(PTA,GW,seed) #BAT.jl currently requires a particular (older) version of DE.jl, which throws annoying warnings relating to depreceated features in Julia 1.9
        setting = :GW
        true_par_values = guess_parameters(PTA,P)


        likelihood = let observations = measurement, PTA = PTA, model = setting, known_params =  true_par_values


            params -> begin
        
                # println("PArAMS are")
                # println(params)
                # println("attempt eval")
                ll_value = KF(observations,PTA,known_params,params,model)
        
                # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
                return ll_value #LogDVal(ll_value)
            end
        end


    # function logl(x)
        
    #     dx1 = x .- μ1
    #     dx2 = x .- μ2
    #     f1 = -dx1' * (inv_σ * dx1) / 2
    #     f2 = -dx2' * (inv_σ * dx2) / 2
    #     return logaddexp(f1, f2)
    # end

    priors = [
        Uniform(9e-8,2e-7)
    ]

    # create the model
    # or model = NestedModel(logl, prior_transform)
    model = NestedModel(likelihood, priors);

    #best_val = likelihood([GW.ω])
   # println(best_val)

   
    println("START THE SAMPLER")

    # create our sampler
    # 2 parameters, 1000 active points, multi-ellipsoid. See docstring
    spl = Nested(1, 50)
    # by default, uses dlogz for convergence. Set the keyword args here
    # currently Chains and Array are support chain_types
    #chain, state = sample(model, spl; dlogz=0.2, param_names=["x"])
    chain, state = sample(model, spl; maxlogl=-6e6, param_names=["x"])

    # optionally resample the chain using the weights
    chain_res = sample(chain, Weights(vec(chain["weights"])), length(chain));

    display(chain_res)


    return chain,state,chain_res

    




end 