



function infer_parameters2(::Type{NF}=Float64;              # number format, use Float64 as default
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


likelihood = let observations = measurement, PTA = PTA, model = setting 


    params -> begin
        ll_value = KF(observations,PTA,params,model)

        # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
        return LogDVal(ll_value)
    end
end






# Gaussian mixture model
σ = 0.1
μ1 = ones(2)
μ2 = -ones(2)
inv_σ = diagm(0 => fill(1 / σ^2, 2))

function logl(x)
    dx1 = x .- μ1
    dx2 = x .- μ2
    f1 = -dx1' * (inv_σ * dx1) / 2
    f2 = -dx2' * (inv_σ * dx2) / 2
    return logaddexp(f1, f2)
end
priors = [
    Uniform(-5, 5),
    Uniform(-5, 5)
]
# or equivalently
prior_transform(X) = 10 .* X .- 5
# create the model
# or model = NestedModel(logl, prior_transform)
model = NestedModel(logl, priors);


println("THE MODEL IS")


















#true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)

true_par_values = guess_parameters(PTA,P)

# prior = NamedTupleDist(
#     a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
#     mu = [-2.0..0.0, 1.0..3.0],
#     sigma = Weibull(1.2, 2)
# )

a = Weibull(1.1, 5000)

println("BBbbbbbbaaaaaa")
println(a)







lval = likelihood(true_par_values)



println("output lval = ", lval)








end 