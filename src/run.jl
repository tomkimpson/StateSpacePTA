"""
solution,model = orbit(NF,kwargs...)

Runs RelativisticDynamics.jl with number format `NF` and any additional parameters in the keyword arguments
`kwargs...`. Any unspecified parameters will use the default values as defined in `src/system_parameters.jl`."""
function UKF(::Type{NF}=Float64;              # number format, use Float64 as default
           kwargs...                        # all additional non-default parameters
           ) where {NF<:AbstractFloat}




    P = SystemParameters(NF=NF;kwargs...) # Parameters
    PTA = setup_PTA(P)
    GW = gw_variables(P.NF,P)

    @info "Hello from StateSpacePTA. You are running with NF = ", " ", P.NF

    seed = P.seed # Integer or nothing 
    state,measurement = create_synthetic_data(PTA,GW,seed)


    plotter(PTA.t,state,measurement,nothing,nothing,4)

    #θ̂ = guess_parameters(PTA,P)
   # model_state_predictions,model_likelihood = kalman_filter(measurement,PTA,θ̂,:GW)
    
    # null_state_predictions,null_likelihood = kalman_filter(measurement,PTA,θ̂,:null)

    # test_statistic = NF(2.0) * (model_likelihood - null_likelihood)


    # output_dictionary = Dict("time" => PTA.t, "state" => state, "measurement" => measurement,
    #                          "TS" => test_statistic,
    #                          "model_predictions" => model_state_predictions, "model_likelihood" => model_likelihood,
    #                          "null_predictions" => null_state_predictions, "null_likelihood" => null_likelihood,)


    # return output_dictionary
end