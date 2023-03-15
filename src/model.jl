

"""
Transition function which takes the state of sigma points and advances
by dt using a Euler step.
"""
function F_function(χ::Matrix{NF},dt::NF) where {NF<:AbstractFloat}





    df = -θ̂.γ #.* (χ 
  
    return χ .+ dt .* df 
end 



"""
Measurement function which takes the state and returns the measurement
"""
function H_function(parameters,t,q) where {NF<:AbstractFloat}

    println("this is the H function")

    @unpack h,ι,δ,α,ψ,ω,Φ0,d = parameters 

    m,n,n̄,Hij = gw_variables(h,ι, δ, α, ψ)

    prefactor,dot_product = gw_prefactor(n̄,q,Hij,ω,d)

    GW_factor = gw_modulation(t, ω,Φ0,prefactor,dot_product)

    return diagm(GW_factor) #make it a 2d matrix
end 



"""
Measurement function which takes the state and returns the measurement, but with zero measurement effects
    i.e. just returns the state

"""
function null_function(χ::Matrix{NF},t::NF,dot_product::Vector{NF},prefactor::Vector{Complex{NF}},ω::NF,Φ0::NF) where {NF<:AbstractFloat}
    return χ 
end 


"""
Returns a Q matrix of size N x N pulsars 
"""
function Q_function(γ::Vector{NF},σp::NF,dt::NF) where {NF<:AbstractFloat}
    value = σp^2 .* exp.(NF(2.0).*γ .* dt) .- NF(1.0) ./ (NF(2.0) .* γ)
    return diagm(value) #https://stackoverflow.com/questions/69609872/how-to-make-a-diagonal-matrix-from-a-vector

end 


function R_function(L::Int, σm::NF) where {NF<:AbstractFloat}

    return diagm(fill(σm^2 ,L)) 

end 