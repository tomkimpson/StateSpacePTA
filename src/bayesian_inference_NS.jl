




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


        known_parameters = set_known_parameters(P)
        #true_par_values = guess_parameters(PTA,P)


        
        psr_priors = set_pulsar_priors(PTA)
        gw_priors = [Uniform(1e-8,1e-6)] #priors on ω
        priors = [gw_priors ; psr_priors]
        prior_names = []

        prefix_result = randstring(12)
        psr_prior_names = [randstring(12) for i in psr_priors]
        prior_names = ["omega" ; psr_prior_names]
        println(prior_names)


    
        likelihood = let observations = measurement, PTA = PTA, model = setting, known_params =  known_parameters

            params -> begin               
                ll_value = KF(observations,
                              PTA,
                              known_params,
                              params)
                return ll_value 
            end
        end



    psr_priors= set_single_pulsar_priors(PTA)
    params = [1e-8;psr_priors]
   # params = Float64.(a)
    #println(typeof(params))
    ll_value,model_state_predictions = KF(measurement,PTA,known_parameters,params)
    println("likelihood = ")
    println(ll_value)


    plotter(PTA.t,state,measurement,model_state_predictions,nothing,5)




     #model = NestedModel(likelihood, priors);
    # spl = Nested(1, 100)
     #chain, state = sample(model, spl; dlogz=1e3, param_names=prior_names)


    # #Define the priors 
    # priors = [
    #     Dirac(1e-7) #priors on ω
    # ]





#     #Define the priors 
#     priors = [
#         Uniform(1e-8,1e-6) #priors on ω
#     ]

# #     # create the model
# #     # or model = NestedModel(logl, prior_transform)
#     model = NestedModel(likelihood, priors);

# #     #best_val = likelihood([GW.ω])
# #    # println(best_val)

   
#     println("START THE SAMPLER")
#     println(model)

#     # create our sampler
#     # 2 parameters, 1000 active points, multi-ellipsoid. See docstring
#     spl = Nested(1, 100)
#     # by default, uses dlogz for convergence. Set the keyword args here
#     # currently Chains and Array are support chain_types
#     chain, state = sample(model, spl; dlogz=1e3, param_names=["x"])
#     #chain, state = sample(model, spl; maxlogl=-6e6, param_names=["x"])
#     # optionally resample the chain using the weights
     #chain_res = sample(chain, Weights(vec(chain["weights"])), length(chain));

#     display(chain_res)


    #return chain,state,chain_res

    




end 