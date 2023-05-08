

import NLSolvers
using ThreadsX
using Base.Threads

using BlackBoxOptim


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
        Φ0 = params[2]

        P1 = SystemParameters(NF=Float64,ω_guess=ω,Φ0_guess=Φ0) 
        θ̂1 = kalman_parameters(PTA,P1)


        ll_value,x_results = KF(measurements,
                                PTA,
                                θ̂1,
                                )

        #@info "optimisation_wrapper function with omega = ", ω,ll_value

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
    x0 = NF.([1e-7,0.10])



    xlower = NF.([1e-8,0.0])
    xupper = NF.([1e-5,1.0])
 
    obj = NLSolvers.ScalarObjective(optimisation_wrapper, 
                                    nothing, nothing, nothing, nothing, nothing,  
                                    himmelblau_batched!, nothing)

    prob = NLSolvers.OptimizationProblem(obj, (xlower, xupper))

    opt = NLSolvers.solve(prob, x0, NLSolvers.ParticleSwarm(n_particles=1000), NLSolvers.OptimizationOptions(maxiter=30))


    #output = NLSolvers.minimizer(opt)

    display(opt)
    println(opt.info.solution)

end 





# https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/multithreaded_optimization.jl

function blackbox(::Type{NF}=Float64;              # number format, use Float64 as default
    kwargs...                        # all additional non-default parameters
    ) where {NF<:AbstractFloat}
    
        @info "Attempting a blackbox optimisation"
    
        #Setup the system, define the true state and the measurement
        state,measurements,PTA,θ̂,P = setup()
    
    
        best_ll_value,best_x_results = KF(measurements,PTA,θ̂)

        println("The target best ll value is:", best_ll_value)


        P1 = SystemParameters(NF=Float64,ω_guess=5.00448e-7,δ_guess=0.925617) 
        θ̂1 = kalman_parameters(PTA,P1)
        best_ll_value2,best_x_results = KF(measurements,PTA,θ̂1)
        println("The target best ll value is:", best_ll_value2)



        #Define function to feed into optimisation 
        #Measurements, 
        optimisation_wrapper = let measurements = measurements, PTA = PTA
    
        params -> begin 
    
            ω=params[1]
            δ = params[2]
    
            P1 = SystemParameters(NF=Float64,ω_guess=ω,δ_guess=δ) 
            θ̂1 = kalman_parameters(PTA,P1)
    
    
            ll_value,x_results = KF(measurements,
                                    PTA,
                                    θ̂1,
                                    )
    
            #@info "optimisation_wrapper function with = ", ω,δ, ll_value
    
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
    
    truths = [θ̂.ω;θ̂.δ]
    x0 = [2e-7;0.30]
    lower_limits = [2e-9;-π/2.0]
    upper_limits = [1e-6;π/2.0]




    res = bboptimize(optimisation_wrapper; SearchRange = [(1e-9,1e-6),(0.0,1.50)],Method = :adaptive_de_rand_1_bin, MaxSteps=75000)

    #print(res)

    
    end 