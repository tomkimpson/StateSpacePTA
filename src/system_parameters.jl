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

end
