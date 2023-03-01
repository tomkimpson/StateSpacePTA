



function create_synthetic_data(
                               pulsars::Pulsars,
                               GW::gravitational_wave) where {NF<:AbstractFloat}


    @unpack q,t = pulsars 
    @unpack Ω,Hij = GW

   
    

    #Create some useful quantities that relate the GW and pulsar variables 
    dot_product = [1.0 .+ dot(Ω,q[i,:]) for i=1:size(q)[1]]                   # Size Npulsars. Is there a vectorised way to do this?
    hbar = [sum([Hij[i,j]*q[k,i]*q[k,j] for i=1:3,j=1:3]) for k=1:size(q)[1]] # Size Npulsars. Is there a vectorised way to do this?
    
    #blob = t * dot_product
    println(size(t))
    println(size(dot_product))

    #time_variation = exp(-1im*Ω*t*dot_product + self.phase_normalisation)

            
            
            
    # GW_factor = np.real(1 - NF(0.5)*h_scalar/dot_product *time_variation*(1 - np.exp(1j*self.omega_GW*d*dot_product/c)))




end 