

"""
Transition function
"""
function F_function(γ::Vector{NF},dt::NF) where {NF<:AbstractFloat}
    value = exp.(-γ.*dt)
    return value
end 

"""
Control function
"""
function T_function(f0::Vector{NF}, ḟ0::Vector{NF},γ::Vector{NF},t::Vector{NF},dt::NF) where {NF<:AbstractFloat}
   
   
   
   
    ḟ0_time = (t.+dt) * ḟ0' #This has shape(n times, n pulsars)
    tmp_value = f0 .+ ḟ0_time'
    value = tmp_value .+ ḟ0.*dt .- exp.(-γ.*dt).*(tmp_value)
    return value #size(npulsars, ntimes)
end 


"""
Returns a Q diagonal matrix as a 1D vector 
"""
function Q_function(γ::Vector{NF},σp::NF,dt::NF) where {NF<:AbstractFloat}
    value = -σp^2 .* ((exp.(-NF(2.0).*γ .* dt) .- NF(1.0)) ./ (NF(2.0) .* γ))
    return value
end 

"""
Returns a R diagonal matrix as a scalar
"""
function R_function(σm::NF) where {NF<:AbstractFloat}
    return σm^2
end 