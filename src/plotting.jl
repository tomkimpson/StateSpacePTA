function plotter(t,states,measurements,psr_index)


    println("You have called the ploter")

    tplot = t / (365*24*3600)
    state_i = states[psr_index,:]
    measurement_i = measurements[psr_index,:]


    plt = plot(tplot,state_i,layout=grid(2,1, heights=(0.5,0.5)), size=(1000,1000),legend=false,link = :x)


    plot!(tplot,measurement_i,subplot=2,legend=false)


    plot!(xlabel="t [years]",subplot=2)
    plot!(ylabel="f [Hz]",subplot=2)
    plot!(ylabel="f [Hz]",subplot=1)



    # plt = plot(t,blob,
    # xlabel="thing3343",
    # ylabel="thing4",
    # legend=false,
    # size = (600, 600)
    # )

    display(plt)

end 