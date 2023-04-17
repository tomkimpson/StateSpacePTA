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
    ω      ::Real = 5e-7      
    Φ0     ::Real = 0.20      
    ψ      ::Real = 2.5 
    cos_ι  ::Real = 0.921060994 #this is cos(0.4)
    δ  ::Real = 1.0
    α :: Real = 1.0
    h :: Real = 1e-2

    #Noise parameters
    σp::Real = 1e-16
    σm::Real = 1e-10


    #Pulsar parameters 
    Npsr::Int64 = 0 #0 is the default for ALL
    γ::Real = 1e-13 # all pulsars have the same \gamma

    #Random seed
    seed::Int64 = 0 #0 is defined to be no seeding - different for each run!



    #Plotting settings
    psr_index::Int64 = 1 #which psr to plot

    
    #Guessed parameters
    #Whilst the GW parameters are used in conjunction with the pulsar parameters to generate synthetic data
    #These parameters are the guesses of those quantities which get fed into the Kalman FILTER

    #d_guess for now take d as known i.e. read from data file. Note that these are vector parameters, one for each pulsar
    #γ_guess ditto
    #n_guess ditto

    ω_guess  ::Real = ω      
    Φ0_guess ::Real = Φ0      
    ψ_guess  ::Real = ψ 
    ι_guess  ::Real = cos_ι
    δ_guess  ::Real = δ
    α_guess :: Real = α
    h_guess :: Real = h

    σp_guess::Real = σp
    σm_guess::Real = σm



end
