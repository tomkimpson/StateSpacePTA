



function create_synthetic_data(
                               pulsars::Pulsars,
                               GW::gravitational_wave) #where {NF<:AbstractFloat}


    @unpack f0,q,t,d,n,γ,σp,σm = pulsars 
    @unpack Ω,Hij,ω,Φ0 = GW

    NF = eltype(t)

    
      
    #Evolve the pulsar frequency 
    f(u,p,t) = -γ.*u .^n
    g(u,p,t) = σp
    tspan = (first(t),last(t))
    prob = SDEProblem(f,g,f0,tspan,tstops=t)
    intrinsic_frequency = solve(prob,EM())
    


    #Create some useful quantities that relate the GW and pulsar variables 
    prefactor,dot_product = gw_prefactor(Ω,q,Hij,ω,d)

   
    #Iterate through time. Really should vectorise all this...
    #But loops fast in Julia...
    f_measured_clean = zeros(NF,size(q)[1],length(t))


    for i =1:length(t)
       ti = t[i]
    
       time_variation = exp.(-1im*ω*ti .*dot_product .+ Φ0)
       GW_factor = real.(NF(1.0) .- prefactor .* time_variation)
    
       f_measured_clean[:,i] = intrinsic_frequency[:,i] .* GW_factor
    end


    f_measured = add_gauss(f_measured_clean, σm, 0.0) #does this do the correct thing?   
    return intrinsic_frequency,f_measured

end 