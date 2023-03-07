

"""
Transition function which takes the state of sigma points and advances
by dt using a Euler step.
"""
function F_function(χ::Matrix{NF},dt::NF,θ̂::GuessedParameters) where {NF<:AbstractFloat}
    df = -θ̂.γ .*χ.^θ̂.n
  
    return χ .+ dt .* df 
end 



"""
Measurement function which takes the state and returns the measurement
"""
function H_function(χ::Matrix{NF},t::NF,dot_product::Vector{NF},prefactor::Vector{Complex{NF}},ω::NF,Φ0::NF) where {NF<:AbstractFloat}
    time_variation = exp.(-1im*ω*t.*dot_product .+ Φ0)
    GW_factor = real(NF(1.0) .- prefactor .* time_variation)
    GW = reshape(GW_factor,(1,size(GW_factor)[1])) #make GW_factor a row vector for operations with Χ(95,47)
    return χ .* GW
end 



"""
Measurement function which takes the state and returns the measurement, but with zero measurement effects
    i.e. just returns the state

"""
function null_function(χ::Matrix{NF},t::NF,dot_product::Vector{NF},prefactor::Vector{Complex{NF}},ω::NF,Φ0::NF) where {NF<:AbstractFloat}
    return χ 
end 



function Q_function(γ::Matrix{NF},n::Matrix{NF},f0::Matrix{NF},dt::NF,σp::NF) where {NF<:AbstractFloat}

    #println("hello from inside Q func")

    coefficient = NF(2) .*γ .*n .*f0.^(n.-1)
    exponential_term = exp.(-coefficient.*dt) .- NF(1.0)
    
    Q = -(σp)^2 .* exponential_term ./ coefficient

    return diagm(Q[1,:]) #https://stackoverflow.com/questions/69609872/how-to-make-a-diagonal-matrix-from-a-vector

end 


function R_function(L::Int, σm::NF) where {NF<:AbstractFloat}

    return diagm(fill(σm^2 ,L)) 

end 