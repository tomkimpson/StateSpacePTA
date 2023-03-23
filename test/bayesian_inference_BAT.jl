



function infer_parameters(::Type{NF}=Float64;              # number format, use Float64 as default
        kwargs...                        # all additional non-default parameters
        ) where {NF<:AbstractFloat}


        P = SystemParameters(NF=NF;kwargs...) # Parameters
        PTA = setup_PTA(P)
        GW = gw_variables(P.NF,P)
    
        @info "Hello from StateSpacePTA. You are running with NF = ", P.NF
    
        seed = P.seed # Integer or nothing 
        state,measurement = create_synthetic_data(PTA,GW,seed) #BAT.jl currently requires a particular (older) version of DE.jl, which throws annoying warnings relating to depreceated features in Julia 1.9
    
        println(size(measurement))


        setting = :GW
        true_par_values = guess_parameters(PTA,P)




    println("hello from the inference module")


    likelihood = let observations = measurement, PTA = PTA, model = setting, known_params =  true_par_values


        params -> begin

            # println("PArAMS are")
            # println(params)
            # println("attempt eval")
            ll_value = KF(observations,PTA,known_params,params,model)

            # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
            return LogDVal(ll_value)
        end
    end





#true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)


# prior = NamedTupleDist(
#     a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
#     mu = [-2.0..0.0, 1.0..3.0],
#     sigma = Weibull(1.2, 2)
# )

a = Uniform(1,10) #logUniform only defined in v025?

#priors = set_priors(NF)


priors = NamedTupleDist(
    ω = Uniform(5e-7, 5e-6),
)



posterior = PosteriorDensity(likelihood, priors)



samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10, nchains = 2)).result






# println("POSTERIOR IS:")
# println(posterior)

l#val = likelihood(blobprior)

#lval = likelihood(true_par_values)



#println("output lval = ", lval)



end 