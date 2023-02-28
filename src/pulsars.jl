


struct Pulsars{NF<:AbstractFloat}

    f :: Vector{NF} 
    d :: Vector{NF}
    γ :: Vector{NF}
    n :: Vector{NF} 
    q :: Matrix{NF} 


end 

function setup_PTA(NF::Type)

    pc = 3e16     # parsec in m


    pulsars = DataFrame(CSV.File("data/NANOGrav_pulsars.csv"))

    f = pulsars[:,"F0"]
    d = pulsars[:,"DIST"]*1e3*pc
    γ = pulsars[:,"gamma"]
    n = pulsars[:,"n"]

    δ = pulsars[:,"DECJD"]
    α = pulsars[:,"RAJD"]

    
    q = unit_vector(π/2.0 .-δ, α)
    println(typeof(q))

    return Pulsars{NF}(f,d,γ,n,q)


end 


function unit_vector(θ::Vector{NF},ϕ::Vector{NF}) where {NF<:AbstractFloat}

   
   qx = sin.(θ) .* cos.(ϕ)
   qy = sin.(θ) .* sin.(ϕ)
   qz = cos.(θ)

   return q =[qx;;qy;;qz] #shape Npulars,3
   
end 