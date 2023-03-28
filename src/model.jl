

"""
Transition function which takes the state of sigma points and advances
by dt using a Euler step.
"""
function F_function(γ::Vector{NF},dt::NF) where {NF<:AbstractFloat}
    value = exp.(-γ.*dt)
    return Diagonal(value) 
end 

function T_function(f0::Vector{NF}, ḟ0::Vector{NF},γ::Vector{NF},t,dt) where {NF<:AbstractFloat}
    return f0 + ḟ0*(t+dt) - exp.(-γ.*dt).*(f0+ḟ0*t)
end 



"""
Measurement function which takes the state and returns the measurement
"""
function H_function(t::NF, ω::NF,Φ0::NF,prefactor::Vector{Float64},dot_product::Vector{Float64}) where {NF<:AbstractFloat}
    GW_factor = gw_modulation(t, ω,Φ0,prefactor,dot_product)
    return Diagonal(GW_factor) #make it a 2d matrix
end 


"""
Don't do anything!
"""
function null_function(t::NF, ω::NF,Φ0::NF,prefactor::Vector{Float64},dot_product::Vector{Float64}) where {NF<:AbstractFloat}
    return Diagonal(fill(NF(1.0) ,length(dot_product))) 
end 


"""
Measurement function which takes the state and returns the measurement, but with zero measurement effects
    i.e. just returns the state

"""
function null_function(χ::Matrix{NF},t::NF,dot_product::Vector{NF},prefactor::Vector{Complex{NF}},ω::NF,Φ0::NF) where {NF<:AbstractFloat}

end 


"""
Returns a Q matrix of size N x N pulsars 
"""
function Q_function(γ::Vector{NF},σp::NF,dt::NF) where {NF<:AbstractFloat}
    value = σp^2 .* ((exp.(NF(2.0).*γ .* dt) .- NF(1.0)) ./ (NF(2.0) .* γ))
    push!(value,0.0) #add a 0
    return Diagonal(value) 
end 

function R_function(L::Int, σm::NF) where {NF<:AbstractFloat}
    return Diagonal(fill(σm^2 ,L)) 
end 