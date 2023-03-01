



function create_synthetic_data(
                               pulsars::Pulsars,
                               GW::gravitational_wave) #where {NF<:AbstractFloat}


    @unpack f0,q,t,d,n,γ,σp,σm = pulsars 
    @unpack Ω,Hij,ω,Φ0 = GW

    NF = eltype(t)

    
      
    #Evolve the pulsar frequency 
    f(u,p,t) = γ.*u .^n
    g(u,p,t) = σp
    tspan = (first(t),last(t))
    prob = SDEProblem(f,g,f0,tspan,tstops=t)
    intrinsic_frequency = solve(prob,EM())#,dt=dt)
    


    #Create some useful quantities that relate the GW and pulsar variables 
    dot_product = [1.0 .+ dot(Ω,q[i,:]) for i=1:size(q)[1]]                   # Size Npulsars. Is there a vectorised way to do this?
    hbar = [sum([Hij[i,j]*q[k,i]*q[k,j] for i=1:3,j=1:3]) for k=1:size(q)[1]] # Size Npulsars. Is there a vectorised way to do this?
    
    ratio = hbar ./ dot_product
    Hcoefficient = NF(1.0) .- exp.(1im*ω.*d.*dot_product)
    prefactor = NF(0.5).*ratio.*Hcoefficient
    

    #Iterate through time. Really should vectorise all this...
    #But loops fast in Julia...
    f_measured_clean = zeros(NF,length(t),size(q)[1])
    for i =1:length(t)
       ti = t[i]
    
       time_variation = exp.(-1im*ω*ti .*dot_product .+ Φ0)
       GW_factor = real.(NF(1.0) .* prefactor .* time_variation)
    
       f_measured[i,:] = intrinsic_frequency[:,i] .* GW_factor
    end


    


    #println(time_variation)
    #end 

    #mayeb can vectorise this, but speed is ok?
    #output array is t x Npulsars 
    #blob = t * dot_product
    # println(size(t))
    # println(size(dot_product))
    # println(size(hbar))

    #blob = reshape([t .* dot_product[i] for i=1:length(dot_product)],(522,47))
    #println(size(blob[1]))
    #for i = 1:size(fot)
    #time_variation = exp(-1im*Ω*t*dot_product + self.phase_normalisation)

            
            
            
    # GW_factor = np.real(1 - NF(0.5)*h_scalar/dot_product *time_variation*(1 - np.exp(1j*self.omega_GW*d*dot_product/c)))




end 