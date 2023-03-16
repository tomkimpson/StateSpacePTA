



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




println("hello from the inference module")


# data = vcat(
#     rand(Normal(-1.0, 0.5), 500),
#     rand(Normal( 2.0, 0.5), 1000)
# )



# hist = append!(Histogram(-2:0.1:4), data)



# function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
#     p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
#     p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
# end




likelihood = let observations = measurement, PTA = PTA, model = setting 


    params -> begin

        #println(i)
        # Log-likelihood for a single bin:
        # function bin_log_likelihood(i)
        #     # Simple mid-point rule integration of fit function `f` over bin:
        #     expected_counts = bin_widths[i] * f(params, bin_centers[i])
        #     logpdf(Poisson(expected_counts), observed_counts[i])
        # end

        # # Sum log-likelihood over bins:
        # idxs = eachindex(observed_counts)
        # ll_value = bin_log_likelihood(idxs[1])
        # for i in idxs[2:end]
        #     ll_value += bin_log_likelihood(i)
        # end

        ll_value = KF(observations,PTA,params,model)

        # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
        return LogDVal(ll_value)
    end
end





#true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)

true_par_values = guess_parameters(PTA,P)



lval = likelihood(true_par_values)



println("output lval = ", lval)








end 