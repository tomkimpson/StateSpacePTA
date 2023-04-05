



function create_synthetic_data(
                               pulsars::Pulsars,
                               GW::GW_Parameters,
                               seed::Int64) 


    @unpack f0, ḟ0,q,t,d,γ,σp,σm,q = pulsars 

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
    
    #Get the modulation factor due to GW 
    @unpack δ,α,ψ,h,cos_ι,ω,Φ0 = GW
    gw_factor = gw_frequency_modulation_factor(δ,α,ψ,h,cos_ι,ω,Φ0,q,d,t)
   
    #The measured frequency without noise 
    f_measured_clean = gw_factor .* intrinsic_frequency
    #The measured frequency with noise 
    f_measured = add_gauss(f_measured_clean, σm, 0.0)    

    return intrinsic_frequency,f_measured
  
end 