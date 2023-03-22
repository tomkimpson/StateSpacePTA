

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
function H_function(χ::Matrix{NF},t::NF,Npulsars::Int64,q::Matrix{NF},d::Vector{NF}) where {NF<:AbstractFloat}


    frequencies = χ[:,1:Npulsars]           # sigma vector of frequencies
    parameters = χ[:,Npulsars+1:Npulsars+7] # sigma vector of frequencies
    J = size(χ)[1] #2L+1
    
    #Ω = parameters[:,6] #size 109 

    h = parameters[:,1]
    ι = parameters[:,2]
    δ = parameters[:,3]
    α = parameters[:,4]
    ψ = parameters[:,5]
    ω = parameters[:,6]
    Φ0= parameters[:,7]

    m,n                 = principal_axes(π/NF(2.0) .- δ,α,ψ) 
    Ω                   = [cross(m[i,:],n[i,:]) for i=1:J]

    hp,hx               = h_amplitudes(h,ι)                                    

    

    Hij_sigma = zeros(NF,3,3,J) #one 3x3 matrix for every sigma vector!
    for i=1:J #for every sigma vector 
        mi = m[i,:]
        ni = n[i,:]

        e_plus  = [mi[j]*mi[k]-ni[j]*ni[k] for j=1:3,k=1:3]
        e_cross = [mi[j]*ni[k]-ni[j]*mi[k] for j=1:3,k=1:3]


        Hij                 = hp[i] .* e_plus .+ hx[i] * e_cross
        
        Hij_sigma[:,:,i] = Hij

    end 
    
    hbar_sigma = zeros(NF,J, Npulsars)
    

    for b=1:J
        Hij = Hij_sigma[:,:,b]
        hbar = [sum([Hij[i,j]*q[k,i]*q[k,j] for i=1:3,j=1:3]) for k=1:size(q)[1]] # Size Npulsars. Is there a vectorised way to do this?
        hbar_sigma[b,:] = hbar
    end 





    #Can we avoid the loop here for performance? 
    dot_product = zeros(NF,J,Npulsars)
    for i=1:J
        Ωi = Ω[i] #this is 3 components x,y,z 

        for k=1:Npulsars
            qi = q[k,:] #this also has 3 components  
            dot_product_i = NF(1.0) + dot(Ωi,qi) #scalar 
            dot_product[i,k] = dot_product_i
        end 


    end 
   


    #println("size hbar: ", size(hbar_sigma))
    #println("size dot product: ", size(dot_product))

    ratio = hbar_sigma ./ dot_product
    #println(size(ratio))
    #println(size(ω))

    Hcoefficient_sigma = zeros(Complex{NF},J,Npulsars)

    for i=1:J
        ωi = ω[i]
        dot_product_i = dot_product[i,:] #all the pulsars of the ith sigma vector       
        Hcoefficient = NF(1.0) .- exp.(1im*ωi.*d.*dot_product_i)
        Hcoefficient_sigma[i,:] = Hcoefficient
   
    end 



    prefactor = NF(0.5).*ratio.*Hcoefficient_sigma

  
    time_variation = exp.(-1im*ω*t.*dot_product .+ Φ0)
    GW_factor = real(NF(1.0) .- prefactor .* time_variation)
  


    #println("Size of Ω:", size(Ω))
    #println("Size of q:", size(q[1,:]))



    #dot_product = [NF(1.0) .+ dot(Ω,q[i,:]) for i=1:size(q)[1]] 


    #frequencies = 




    #GW = reshape(GW_factor,(1,size(GW_factor)[1])) #make GW_factor a row vector for operations with Χ(95,47)
    #println("EXIT")
    return frequencies .* GW_factor
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
    value = exp.(NF(2.0).*γ .* dt) .- NF(1.0) ./ (NF(2.0) .* γ)
    return diagm(value) #https://stackoverflow.com/questions/69609872/how-to-make-a-diagonal-matrix-from-a-vector

end 


function R_function(L::Int, σm::NF) where {NF<:AbstractFloat}

    return diagm(fill(σm^2 ,L)) 

end 