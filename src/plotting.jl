function plotter(t,states,measurements,model_state_predictions,null_state_predictions,psr_index)


    #println("You have called the plot function")

    tplot = t / (365*24*3600)
    state_i = states[psr_index,:]
    measurement_i = measurements[psr_index,:]



    #Plot the state
    plt = plot(tplot,state_i,layout=grid(2,2, heights=(0.5,0.5)), size=(1000,1000),legend=false,link = :x)
    plot!(ylabel="f STATE [Hz]",subplot=1)

    @info "The variance in the state is:", var(state_i)
    @info "Final f - Initial f = ", last(state_i) - first(state_i)
    @info "Δf between steps is = ", state_i[2] - first(state_i)


    #Plot the measurement
    plot!(tplot,measurement_i,subplot=3,legend=false,linecolor=:green)
    plot!(ylabel="f MEASURED [Hz]",subplot=3)




    #Plot the predictions
    if model_state_predictions != nothing
        model_prediction_i = model_state_predictions[:,psr_index] #psr index now indexes second axis sicne state predictions are a different shape! Annoying!
        println("final prediction: ", last(model_prediction_i))
        println("final mesurement: ", last(measurement_i))

        plot!(tplot,state_i,subplot=2,label="State")   
        plot!(tplot,model_prediction_i,subplot=2,label="Prediction")    
        plot!(ylabel="f STATE [Hz]",subplot=2)
 
        
        plot!(tplot,state_i,subplot=4,label="State")   
        plot!(ylabel="f STATE [Hz]",subplot=4)
  
     end 





        #null_prediction_i  = null_state_predictions[:,psr_index] #psr index now indexes second axis sicne state predictions are a different shape! Annoying!
        #plot!(tplot,null_prediction_i,subplot=4,label="Prediction")   




    #Some plotting config
    plot!(xlabel="t [years]",subplot=3)
    plot!(xlabel="t [years]",subplot=4)



    # plt = plot(t,blob,
    # xlabel="thing3343",
    # ylabel="thing4",
    # legend=false,
    # size = (600, 600)
    # )

    display(plt)

end 