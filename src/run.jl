"""
solution,model = run_all(NF,kwargs...)

Runs StateSpacePTA.jl with number format `NF` and any additional parameters in the keyword arguments
`kwargs...`. Any unspecified parameters will use the default values as defined in `src/system_parameters.jl`."""
function run_all(::Type{NF}=Float64;              # number format, use Float64 as default
           kwargs...                        # all additional non-default parameters
           ) where {NF<:AbstractFloat}


    state,measurements,PTA,θ̂,P = setup(NF=NF;kwargs...)
    
    model_likelihood,model_predictions = KF(measurements,PTA,θ̂)

    return model_likelihood,model_predictions

end




function setup(::Type{NF}=Float64;              # number format, use Float64 as default
    kwargs...                        # all additional non-default parameters
    ) where {NF<:AbstractFloat}


    P = SystemParameters(NF=NF;kwargs...) # Parameters
    PTA = setup_PTA(P)
    GW_parameters = gw_variables(P)

    #@info "Hello from StateSpacePTA. You are running with NF = ", P.NF, " and a GW strain h = ", P.h
    state,measurement = create_synthetic_data(PTA,GW_parameters,P.seed) 

    θ̂ = kalman_parameters(PTA,P)

    return state,measurement,PTA,θ̂,P


end 










#     #null_likelihood,null_predictions = KF(measurements,PTA,θ̂,:null)

#    # @info "LogLikelihoods are: ", model_likelihood, " for the H1 and ", null_likelihood, " for H0"
#     #@info "This gives a Bayes factor of: ", model_likelihood - null_likelihood

#     plotter(PTA.t,state,measurements,model_predictions,null_predictions,P.psr_index) #Plot the predictions,

