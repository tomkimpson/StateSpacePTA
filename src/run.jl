"""
solution,model = orbit(NF,kwargs...)

Runs StateSpacePTA.jl with number format `NF` and any additional parameters in the keyword arguments
`kwargs...`. Any unspecified parameters will use the default values as defined in `src/system_parameters.jl`."""
function KalmanFilter(::Type{NF}=Float64;              # number format, use Float64 as default
           kwargs...                        # all additional non-default parameters
           ) where {NF<:AbstractFloat}



    @info "Hello from StateSpacePTA. You are running with NF = ", NF
    P   = SystemParameters(NF=NF;kwargs...) # Parameters
    PTA = setup_PTA(P)
    GW  = gw_variables(P.NF,P)
    seed = P.seed # Integer or nothing 
    state,measurement = create_synthetic_data(PTA,GW,seed) 
    #plotter(PTA.t,state,measurement,nothing,nothing,4) #Plot the states if you want

    θ̂ = guess_parameters(PTA,P)


   

    omega_guess = [1e-7]
    likelihood = KF(measurement,PTA,θ̂)






    #model_state_predictions,model_likelihood = EKF(measurement,PTA,θ̂,:GW)

    #model_state_predictions,model_likelihood = UKF(measurement,PTA,:GW)

    #plotter(PTA.t,state,measurement,model_state_predictions,nothing,5)

 #infer_parameters()

    #println(θ̂)

    #model_state_predictions,model_likelihood = KF(measurement,PTA,θ̂,:GW)

    
    # null_state_predictions,null_likelihood = kalman_filter(measurement,PTA,θ̂,:null)

    # test_statistic = NF(2.0) * (model_likelihood - null_likelihood)


    # output_dictionary = Dict("time" => PTA.t, "state" => state, "measurement" => measurement,
    #                          "TS" => test_statistic,
    #                          "model_predictions" => model_state_predictions, "model_likelihood" => model_likelihood,
    #                          "null_predictions" => null_state_predictions, "null_likelihood" => null_likelihood,)


    # return output_dictionary
end




function setup(::Type{NF}=Float64;              # number format, use Float64 as default
    kwargs...                        # all additional non-default parameters
    ) where {NF<:AbstractFloat}


    P = SystemParameters(NF=NF;kwargs...) # Parameters
    PTA = setup_PTA(P)
    GW = gw_variables(P.NF,P)

    @info "Hello from StateSpacePTA. You are running with NF = ", P.NF

    seed = P.seed # Integer or nothing 
    state,measurement = create_synthetic_data(PTA,GW,seed) #BAT.jl currently requires a particular (older) version of DE.jl, which throws annoying warnings relating to depreceated features in Julia 1.9

    θ̂ = guess_parameters(PTA,P)


    return measurement,PTA,θ̂


end 