



function create_synthetic_data(
                               pulsars::Pulsars,
                               GW::gravitational_wave,
                               seed::Int64) 


    @unpack f0, ḟ0,q,t,d,γ,σp,σm = pulsars 
    @unpack n̄,Hij,ω,Φ0 = GW



    NF = eltype(t)
        
    if seed == 0
        Random.seed!()
    else
      Random.seed!(seed)
    end 

    
    #Evolve the pulsar frequencyz
    f(du,u,p,t) = (du .= -γ.*u .+ γ.*(f0 .+ ḟ0*t) .+ ḟ0)
    g(du,u,p,t) = (du .= σp) 
    noise = WienerProcess(0., 0.) # WienerProcess(t0,W0) where t0 is the initial value of time and W0 the initial value of the process
   
    tspan = (first(t),last(t))
    prob = SDEProblem(f,g,f0,tspan,tstops=t,noise=noise)
    intrinsic_frequency = solve(prob,EM())
    
    #Create some useful quantities that relate the GW and pulsar variables 
    prefactor,dot_product = gw_prefactor(n̄,q,Hij,ω,d)

   
    #Iterate through time. Really should vectorise all this...
    #But loops fast in Julia...
    f_measured_clean = zeros(NF,size(q)[1],length(t))


   # for i =1:length(t)


      for i =1:length(t)
        ti = t[i]
     
        time_variation = exp.(-1im*ω*ti .*dot_product .+ Φ0)
        GW_factor = real.(NF(1.0) .- prefactor .* time_variation)
     
        f_measured_clean[:,i] = intrinsic_frequency[:,i] .* GW_factor
     end







    #    GW_factor = gw_modulation(t[i], ω,Φ0,prefactor,dot_product)
    #    f_measured_clean[:,i] = intrinsic_frequency[:,i] .* GW_factor
    # end


    f_measured = add_gauss(f_measured_clean, σm, 0.0) #does this do the correct thing?   
    return intrinsic_frequency,f_measured

end 