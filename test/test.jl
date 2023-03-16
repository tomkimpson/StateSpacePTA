
using DifferentialEquations, Noise, Plots 

#Vector of parameters
γ = [1e-13, 1e-14, 1e-15]
a = [300,400,500]
b = [-1e-15,2e-15,3e-15]

#Initial conditions on y 
y0 = [300,400,500]

σ = 1e-20

#Define the ODEs 

f(du,u,p,t) = (du .= -γ.*u .+ γ.*(a .+ b*t) .+ b)
g(du,u,p,t) = (du .= σ)

#Specify the noise process to use 
noise = WienerProcess(0., 0.) # WienerProcess(t0,W0) where t0 is the initial value of time and W0 the initial value of the process

#Time over which to integrate 
t = range(0.0, 3e8, length=500)
tspan = (first(t),last(t))

#Setup SDE problem and solve
prob = SDEProblem(f,g,y0,tspan,tstops=t,noise=noise)
solution = solve(prob,EM())


i = 1
solution_i = solution[i,:]
plt = plot(t,solution_i)
display(plt)
