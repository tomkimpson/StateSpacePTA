


struct Pulsars{NF<:AbstractFloat}

    f0 :: Vector{NF} 
    d :: Vector{NF}
    γ :: Vector{NF}
    n :: Vector{NF} 
    q :: Matrix{NF} 
    t :: Vector{NF}

    # Noise parameters
    σp::NF 
    σm::NF

    dt::NF 

end 

function setup_PTA(P::SystemParameters)

    pc = 3e16     # parsec in m
    c = 299792458 #speed of light in m/s

    pulsars = DataFrame(CSV.File("data/NANOGrav_pulsars.csv"))

    f = pulsars[:,"F0"]
    d = pulsars[:,"DIST"]*1e3*pc/c #this is in units of s^-1
    γ = pulsars[:,"gamma"]
    n = pulsars[:,"n"]

    δ = pulsars[:,"DECJD"]
    α = pulsars[:,"RAJD"]

    
    q = unit_vector(π/2.0 .-δ, α)
    
    step_seconds = P.cadence * 24*3600 #from days to step_seconds
    end_seconds = P.T * 365*24*3600 #from years to second
    t = collect(0:step_seconds:end_seconds)
   

    return Pulsars{P.NF}(f,d,γ,n,q,t,P.σp, P.σm,step_seconds) #convert to type NF 


end 


function unit_vector(θ::Vector{NF},ϕ::Vector{NF}) where {NF<:AbstractFloat}

   
   qx = sin.(θ) .* cos.(ϕ)
   qy = sin.(θ) .* sin.(ϕ)
   qz = cos.(θ)

   return [qx;;qy;;qz] #shape Npulars,3
   
end 