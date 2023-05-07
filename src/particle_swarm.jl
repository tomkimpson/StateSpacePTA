




function particle_swarm(::Type{NF}=Float64;              # number format, use Float64 as default
kwargs...                        # all additional non-default parameters
) where {NF<:AbstractFloat}



    println("attempting a particle swarm optimisation")

    state,measurements,PTA,θ̂,P = setup()
    #model_likelihood,model_predictions = KF(measurements,PTA,θ̂);

        optimisation_wrapper = let measurements = measurements, PTA = PTA, P=P

        params -> begin 


            α=params[1]
            δ = params[2]

            P1 = SystemParameters(NF=Float64,α_guess=α,δ_guess=δ) 
            θ̂1 = kalman_parameters(PTA,P1)


            ll_value,x_results = KF(measurements,
                                    PTA,
                                    θ̂1,
                                    )
            return -ll_value
        end
        end


    truths = [θ̂.α;θ̂.δ]
    x0 = [0.8;0.8]
    lower_limits = [0.0;0.0]
    upper_limits = [π/2.0;π/2.0]



    nparticles = Int64(100*length(truths))
    opt = optimize(optimisation_wrapper, 
                   x0, 
                   ParticleSwarm(lower_limits,upper_limits,nparticles,threaded=true),
                   Optim.Options(show_trace=true, store_trace=true,iterations = Int64(1e4),show_every=100))



#opt = ParticleSwarm(optimisation_wrapper, x0)


    println("Opt completed")
    display(opt)
    output = Optim.minimizer(opt)
    # #

    for i in 1:length(output)
    println(i, " ", truths[i], " ", output[i])
    end





end 