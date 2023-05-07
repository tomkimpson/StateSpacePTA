

import NLSolvers
using ThreadsX
using Base.Threads



function particle_swarm_v2(::Type{NF}=Float64;              # number format, use Float64 as default
kwargs...                        # all additional non-default parameters
) where {NF<:AbstractFloat}

    @info "Attempting a multithreaded PSO"

    #Setup the system, define the true state and the measurement
    state,measurements,PTA,θ̂,P = setup()


    #Define function to feed into optimisation 
    #Measurements, 
    optimisation_wrapper = let measurements = measurements, PTA = PTA

    params -> begin 

        ω=params[1]

        P1 = SystemParameters(NF=Float64,ω_guess=ω) 
        θ̂1 = kalman_parameters(PTA,P1)


        ll_value,x_results = KF(measurements,
                                PTA,
                                θ̂1,
                                )

       # @info "optimisation_wrapper function with omega = ", ω,ll_value

        return -ll_value
    end
    end

    # function himmelblau!(x)
    #        println(Threads.threadid())
    #        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    #        return fx
    #    end

    function himmelblau_batched!(F, X)
           ThreadsX.map!(optimisation_wrapper, F, X)
           return F
       end

    #x0 = big.([3.0,1.0])
    x0 = NF.([1e-7])
    obj = NLSolvers.ScalarObjective(optimisation_wrapper, 
                                    nothing, nothing, nothing, nothing, nothing,  
                                    himmelblau_batched!, nothing)

    prob = NLSolvers.OptimizationProblem(obj, (1e-8, 1e-5))

    opt = NLSolvers.solve(prob, x0, NLSolvers.ParticleSwarm(n_particles=200), NLSolvers.OptimizationOptions(maxiter=10))


    #output = NLSolvers.minimizer(opt)

    display(opt)
    println(opt.info.solution)

end 