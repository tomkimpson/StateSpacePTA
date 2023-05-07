

struct GW_Parameters{NF<:AbstractFloat}
    ω      ::NF
    Φ0     ::NF    
    ψ      ::NF
    cos_ι  ::NF
    δ  ::NF
    α :: NF
    h :: NF


end 

"""
Cast the user-defined GW parameters to the correct number format
"""
function gw_variables(P::SystemParameters) 
    return GW_Parameters{P.NF}(P.ω,P.Φ0,P.ψ,P.cos_ι, P.δ,P.α,P.h)
end 




"""
Calculate the modulation factor that maps from state space to measurement space 
Accepts as arguments gw_parameters, pulsar locations (q), pulsar distances (d) and observation times (t).
"""
function gw_frequency_modulation_factor(δ::NF,α::NF,ψ::NF,h::NF,cos_ι::NF,ω::NF,Φ0::NF,q::Matrix{NF},d::Vector{NF},t::Vector{NF}) where {NF<:AbstractFloat}


   


    #
    m,n                  = principal_axes(NF(π/2.0) - δ,α,ψ)  
    gw_direction         = cross(m,n)
    dot_product = NF(1.0) .+ q * gw_direction

    e_plus              = [m[i]*m[j]-n[i]*n[j] for i=1:3,j=1:3]
    e_cross             = [m[i]*n[j]-n[i]*m[j] for i=1:3,j=1:3]
    hp,hx               = h_amplitudes(h,cos_ι)   
    Hij                 = hp * e_plus + hx * e_cross
    hbar         = [sum([Hij[i,j]*q[k,i]*q[k,j] for i=1:3,j=1:3]) for k=1:size(q)[1]] # Size Npulsars. Is there a vectorised way to do this?
    prefactor    = NF(0.5).*(hbar ./ dot_product).*(NF(1.0) .- cos.(ω.*d.*dot_product))
    tensor_product = t * dot_product' #this has shape(n times, n pulsars)
    time_variation = cos.(-ω*tensor_product .+ Φ0)
    GW_factor = NF(1.0) .- prefactor' .* time_variation

    
    return GW_factor'
    #@debug @assert size(dot_product)==length(q) # assert is not called unless 

    
end 




"""
Given the location of the GW source (θ, ϕ) and the polarisation angle (ψ)
determine the principal axes of propagation
"""
function principal_axes(θ::NF,ϕ::NF,ψ::NF) where {NF<:AbstractFloat}


    m1 = sin(ϕ)*cos(ψ) - sin(ψ)*cos(ϕ)*cos(θ)
    m2 = -(cos(ϕ)*cos(ψ) + sin(ψ)*sin(ϕ)*cos(θ))
    m3 = sin(ψ)*sin(θ)
    m = [m1,m2,m3]

    n1 = -sin(ϕ)*sin(ψ) - cos(ψ)*cos(ϕ)*cos(θ)
    n2 = cos(ϕ)*sin(ψ) - cos(ψ)*sin(ϕ)*cos(θ)
    n3 = cos(ψ)*sin(θ)
    n = [n1,n2,n3]

    return m,n


end 

"""
Given the strain h and the inclination ι, get the h+ and hx components
"""
function h_amplitudes(h::NF,cos_ι::NF) where {NF<:AbstractFloat}


    hplus = h*(1.0 + cos_ι^2)
    hcross = h*(-2.0*cos_ι)

    return hplus,hcross

end 
