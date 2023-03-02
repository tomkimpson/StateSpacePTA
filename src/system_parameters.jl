"""
    P = Parameters(kwargs...)

A struct to hold all model parameters that may be changed by the user.
The struct uses keywords such that default values can be changed at creation.
The default values of the keywords define the default model setup.
"""
@with_kw struct SystemParameters

    # NUMBER FORMATS
    NF::DataType       # Number format. Default is defined in orbit.jl

    #Observation parameters
    T::Real = 10.0        # how long to integrate for in years
    cadence::Real = 7.0   # sampling interval in days

    #GW parameters
    ω  ::Real = 1e-7      
    Φ0 ::Real = 0.20      
    ψ  ::Real = 2.5 
    ι  ::Real = 0.0
    δ  ::Real = 0.0
    α :: Real = 1.0
    h :: Real = 1e-8

    #Noise parameters
    σp::Real = 1e-13
    σm::Real = 1e-13

    
    #Guessed parameters
    #Whilst the GW parameters are used in conjunction with the pulsar parameters to generate synthetic data
    #These parameters are the guesses of those quantities which get fed into the Kalman FILTER

    #d_guess for now take d as known i.e. read from data file. Note that these are vector parameters, one for each pulsar
    #γ_guess ditto
    #n_guess ditto

    ω_guess  ::Real = ω      
    Φ0_guess ::Real = Φ0      
    ψ_guess  ::Real = ψ 
    ι_guess  ::Real = ι
    δ_guess  ::Real = δ
    α_guess :: Real = α
    h_guess :: Real = h

    σp_guess::Real = σp
    σm_guess::Real = σm



end
