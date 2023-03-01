





struct gravitational_wave{NF<:AbstractFloat}


    m :: Vector{NF}  
    n :: Vector{NF}  
    Ω :: Vector{NF}  

    Hij :: Matrix{NF} 





end 


function gw_variables(P::SystemParameters)

    m,n                 = principal_axes(π/2.0 - P.δ,P.α,P.ψ)    
    Ω                   = cross(m,n)            
    
    hp,hx               = h_amplitudes(P.h,P.ι)                                    
    e_plus              = [m[i]*m[j]-n[i]*n[j] for i=1:3,j=1:3]
    e_cross             = [m[i]*n[j]-n[i]*m[j] for i=1:3,j=1:3]
    
    Hij                 = hp .* e_plus .+ hx * e_cross

    
    return gravitational_wave{P.NF}(m,n,Ω,Hij)



    #x = 0:0.1:1
    #println(x)
    #blob = [m[i]*m[j] -n[i]*n[j] for i=1:3, j=1:3]


    # Hij = self.hp * eplus + self.hx * ecross



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
function h_amplitudes(h::NF,ι::NF) where {NF<:AbstractFloat}


    hplus = h*(1.0 + cos(ι)^2)
    hcross = h*(-2.0*cos(ι))

    return hplus,hcross

end 



function gw_modulation()






end 