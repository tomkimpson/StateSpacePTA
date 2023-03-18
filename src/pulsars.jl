


struct Pulsars{NF<:AbstractFloat}

    f0 :: Vector{NF} 
    ḟ0 :: Vector{NF}
    d :: Vector{NF}
    γ :: Vector{NF}
    #n :: Vector{NF} 
    q :: Matrix{NF} 
    t :: Vector{NF}

    # Noise parameters
    σp::NF 
    σm::NF

    dt::NF 

end 

function setup_PTA(P::SystemParameters)

    pc = 3e16     # parsec in m
    c = 3e8 #speed of light in m/s

    load_file = pkgdir(StateSpacePTA, "data", "NANOGrav_pulsars.csv")

    pulsars = DataFrame(CSV.File(load_file))
    pulsars = first(pulsars,10)#.sample(2) 


    f = pulsars[:,"F0"]
    ḟ = pulsars[:,"F1"] 
    d = pulsars[:,"DIST"]*1e3*pc/c #this is in units of s^-1
    γ = 1e-13 .* (pulsars[:,"gamma"] ./ pulsars[:,"gamma"]) #for every pulsar let γ be 1e-13
   

    δ = pulsars[:,"DECJD"]
    α = pulsars[:,"RAJD"]

    
    q = unit_vector(π/2.0 .-δ, α)
    
    step_seconds = P.cadence * 24*3600 #from days to step_seconds
    end_seconds = P.T * 365*24*3600 #from years to second
    t = collect(0:step_seconds:end_seconds)
   

    return Pulsars{P.NF}(f,ḟ,d,γ,q,t,P.σp, P.σm,step_seconds) #convert to type NF 


end 


function unit_vector(θ::Vector{NF},ϕ::Vector{NF}) where {NF<:AbstractFloat}

   
   qx = sin.(θ) .* cos.(ϕ)
   qy = sin.(θ) .* sin.(ϕ)
   qz = cos.(θ)

   return [qx;;qy;;qz] #shape Npulars,3
   
end 