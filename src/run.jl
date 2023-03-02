





"""
solution,model = orbit(NF,kwargs...)

Runs RelativisticDynamics.jl with number format `NF` and any additional parameters in the keyword arguments
`kwargs...`. Any unspecified parameters will use the default values as defined in `src/system_parameters.jl`."""
function UKF(::Type{NF}=Float64;              # number format, use Float64 as default
           kwargs...                        # all additional non-default parameters
           ) where {NF<:AbstractFloat}


println("Hello from StateSpacePTA. You are running with NF = ", " ", NF)


P = SystemParameters(NF=NF;kwargs...) # Parameters

println(typeof(P))
PTA = setup_PTA(P)
GW = gw_variables(P.NF,P)


state,measurement = create_synthetic_data(PTA,GW)





θ̂ = guess_parameters(PTA,P)
kalman_filter(measurement,PTA,θ̂)

println("Plotter")
psr_index = 1
plotter(PTA.t,state,measurement,psr_index)

# #A = rand(1.:9.,6,4)
# A = Array{NF}([1 2 3; 4 1 6; 7 8 1])
# Q,R = qr(A);


# df = DataFrame(CSV.File("data/NANOGrav_pulsars.csv"))


# blob = Matrix(df)

# f = NF.(blob[:,3])

# println(f)




# println(df)
# println(size(blob))
# println(blob[2,:])
#A = sparse([NF(1.0), 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])
# println(eltype(A))



# QR(A)
# Setup all system parameters, universal constants etc.
#P = SystemParameters(NF=NF;kwargs...) # Parameters
# bounds_checks(P)                      # Check all parameters are reasonable
# C = Constants(P)                      # Constants
# M = Model(P,C)                        # Pack all of the above into a single *Model struct 

# #Initial conditions 
# initialization = initial_conditions(M)

# #Evolve in time
# solution = timestepping(initialization, M)

# return solution, M

end