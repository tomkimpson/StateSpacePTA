



using BlackBoxOptim

function black_box(::Type{NF}=Float64;              # number format, use Float64 as default
kwargs...                        # all additional non-default parameters
) where {NF<:AbstractFloat}



    println("attempting a black box optimisation")

    state,measurements,PTA,θ̂,P = setup()
  
        optimisation_wrapper = let measurements = measurements, PTA = PTA, P=P

        params -> begin 


            ω=params[1]
            δ = params[2]


            #println(ω, " ", δ)

            P1 = SystemParameters(NF=Float64,ω_guess=ω,δ_guess=δ) 
            θ̂1 = kalman_parameters(PTA,P1)


            ll_value,x_results = KF(measurements,
                                    PTA,
                                    θ̂1,
                                    )
            return -ll_value
        end
        end


    truths = [θ̂.ω;θ̂.δ]
    x0 = [2e-7;0.30]
    lower_limits = [2e-9;-π/2.0]
    upper_limits = [1e-6;π/2.0]




    res = bboptimize(optimisation_wrapper; SearchRange = [(4e-7,6e-7 ),(0.9,1.1)],Method = :de_rand_1_bin, MaxSteps=50000)



    # nparticles = Int64(10*length(truths))
    # opt = optimize(optimisation_wrapper, 
    #                x0, 
    #                ParticleSwarm(lower_limits,upper_limits,nparticles),
    #                Optim.Options(show_trace=true, store_trace=true,iterations = Int64(1e4),show_every=100))



#opt = ParticleSwarm(optimisation_wrapper, x0)


    println("Opt completed")
    # display(opt)
    # output = Optim.minimizer(opt)
    # # #

    # for i in 1:length(output)
    # println(i, " ", truths[i], " ", output[i])
    # end





end 