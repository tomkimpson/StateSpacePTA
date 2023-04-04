




function parameter_estimation(::Type{NF}=Float64;              # number format, use Float64 as default
                              kwargs...                        # all additional non-default parameters
                              ) where {NF<:AbstractFloat}


    println("hello from parameter_estimation()")

    state,measurements,PTA,θ̂,P = setup(NF=NF;kwargs...)
    priors,prior_names         = set_priors(θ̂)




    println(θ̂)


    # optimisation_wrapper let observations = measurements, PTA = PTA,θ=θ̂

    #     ω -> begin

    #     θ̂ = guess_parameters(PTA,P)

    
    #     end



    # end 


    optimisation_wrapper = let observations = measurements, PTA = PTA,P=P

        params -> begin 
            P1 = SystemParameters(NF=NF;ω=params[1]) 

            #f = PTA.f0
            f = params[2:end] 
            #set_of_params = guess_parameters(PTA,P1)
            set_of_params = guess_parameters(f,PTA,P1)
            ll_value,x_results = KF(observations,
                                    PTA,
                                    set_of_params,
                                    :GW)

            #println(P1.ω,"  ", -ll_value)
            return -ll_value
        end
    end


   # 327.8470205611185
    

   f = PTA.f0
   lower = f.*0.98
   upper = f.*1.02


   truths = [θ̂.ω ; f]
   x0 = [2e-7;lower]
   lower_limits = [2e-9;lower]
   upper_limits = [1e-6;upper]

   #push!(x0,lower)# + lower 

   println(x0)
   #println(typeof(length(truths)*10))

#     x0 = [1e-7,327.1]
#opt = optimize(optimisation_wrapper, x0, ParticleSwarm([1e-9,327.0],[1e-6,328.0],10))

    nparticles = Int64(10*length(truths))
    opt = optimize(optimisation_wrapper, x0, ParticleSwarm(lower_limits,upper_limits,nparticles),Optim.Options(show_trace=false, iterations = Int64(1e4)))
    #opt = ParticleSwarm(optimisation_wrapper, x0)


    println("Opt completed")
    display(opt)
    output = Optim.minimizer(opt)
# #
    
    for i in 1:length(output)
        println(i, " ", truths[i], " ", output[i])
    end


#display(output)

   

    # likelihood = let observations = measurements, PTA = PTA

    #     params -> begin 
    #         ll_value,x_results = KF(observations,
    #                                 PTA,
    #                                 params,
    #                                 :GW)
    #         return ll_value
    #     end
    # end

   

    
    
    # model = NestedModel(likelihood, priors)
    # spl = Nested(length(priors), 400) #X parameters, Y active points, multi-ellipsoid. See docstring
    # @info "Number of parameters in the prior is: ", length(priors)

    # display(priors)

    # chain, state = sample(model, spl; dlogz=0.2, param_names=prior_names)

    # # optionally resample the chain using the weights
    # chain_res = sample(chain, Weights(vec(chain["weights"])), length(chain))

    # return chain,state,chain_res


end 