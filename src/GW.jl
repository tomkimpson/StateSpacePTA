

struct gravitational_wave{NF<:AbstractFloat}


    m :: Vector{NF}  
    n :: Vector{NF}  
    Ω :: Vector{NF}  

    Hij :: Matrix{NF} 


    ω  ::NF   
    Φ0 ::NF

end 


function gw_variables(NF,P) #P is either a SystemParameters object or a GuessedParameters object 

    m,n                 = principal_axes(π/2.0 - P.δ,P.α,P.ψ)    
    Ω                   = cross(m,n)            
    
    hp,hx               = h_amplitudes(P.h,P.ι)                                    
    e_plus              = [m[i]*m[j]-n[i]*n[j] for i=1:3,j=1:3]
    e_cross             = [m[i]*n[j]-n[i]*m[j] for i=1:3,j=1:3]
    
    Hij                 = hp .* e_plus .+ hx * e_cross
    
    return gravitational_wave{NF}(m,n,Ω,Hij,P.ω,P.Φ0)

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




function gw_prefactor(Ω:: Vector{NF},q::Matrix{NF},Hij::Matrix{NF},ω::NF, d::Vector{NF}) where {NF<:AbstractFloat}

    dot_product = [NF(1.0) .+ dot(Ω,q[i,:]) for i=1:size(q)[1]] 
    hbar = [sum([Hij[i,j]*q[k,i]*q[k,j] for i=1:3,j=1:3]) for k=1:size(q)[1]] # Size Npulsars. Is there a vectorised way to do this?


    #println("dot product:  ", dot_product)
    #println("hbar :  ", hbar)


    ratio = hbar ./ dot_product
    Hcoefficient = NF(1.0) .- exp.(1im*ω.*d.*dot_product)
    prefactor = NF(0.5).*ratio.*Hcoefficient

    #println("Hcoeff :  ", Hcoefficient[1], "BREAK ",1im,"BREAK ",ω,"BREAK  ",d[1],"BREAK  ",dot_product[1])


    return prefactor,dot_product

end 


function gw_modulation()






end 