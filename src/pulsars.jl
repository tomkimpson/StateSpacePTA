


struct Pulsars{NF<:AbstractFloat}

    f0 :: Vector{NF} 
    ḟ0 :: Vector{NF}
    d :: Vector{NF}
    γ :: Vector{NF}
 
    q :: Matrix{NF} 
    t :: Vector{NF}

    # Noise parameters
    σp::NF 
    σm::NF

    dt::NF 

end 

function setup_PTA(P::SystemParameters)

    #Define some global constants 
    pc = 3e16     # parsec in m
    c = 3e8       # speed of light in m/s


    #Load the pulsars CSV 
    load_file = pkgdir(StateSpacePTA, "data", "NANOGrav_pulsars.csv")
    pulsars = DataFrame(CSV.File(load_file))
    
    if P.Npsr != 0 #i.e. if 0 then just select all the pulsars in the PTA
        pulsars = first(pulsars,P.Npsr) #else select the first N pulsars 
       # @info "The number of pulsars selected for this problem is: ", P.Npsr

    #else
       # @info "All pulsars selected"
    end 

    #Define the pulsar parameters 
    f = pulsars[:,"F0"]
    ḟ = pulsars[:,"F1"] 
    d = pulsars[:,"DIST"]*1e3*pc/c #this is in units of s^-1
    γ = fill(P.γ ,length(f)) 
    δ = pulsars[:,"DECJD"]
    α = pulsars[:,"RAJD"]
    q = unit_vector(π/2.0 .-δ, α)

    
    #Timing variables 
    step_seconds = P.cadence * 24*3600       # from days to seconds
    end_seconds = P.T * 365*24*3600          # from years to seconds
    t = collect(0:step_seconds:end_seconds)  # time range to integrate over 
   
    return Pulsars{P.NF}(f,ḟ,d,γ,q,t,P.σp, P.σm,step_seconds) #convert to type NF 


end 


function unit_vector(θ::Vector{NF},ϕ::Vector{NF}) where {NF<:AbstractFloat}

   
   qx = sin.(θ) .* cos.(ϕ)
   qy = sin.(θ) .* sin.(ϕ)
   qz = cos.(θ)

   return [qx;;qy;;qz] # shape (Npulars,3)
   
end 