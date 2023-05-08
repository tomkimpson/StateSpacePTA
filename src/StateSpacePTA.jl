module StateSpacePTA

#Imports 
import Parameters: @with_kw, @unpack

using CSV,DataFrames, LinearAlgebra, Random, Noise, Plots, Statistics, IntervalSets,Distributions,DifferentialEquations


import StatsBase: Histogram #just for testing, can remove later

import Suppressor: @suppress_err


using TensorOperations

using Optim

using ValueShapes

# #extras for NEstedSamplers
using NestedSamplers,AbstractMCMC
using StatsFuns: logaddexp
using StatsBase: sample, Weights


AbstractMCMC.setprogress!(true)
Random.seed!(8452)

#Exports
export particle_swarm_v2,blackbox, setup, KF, run_all, plotter,particle_swarm,SystemParameters,kalman_parameters,black_box #KalmanFilter,infer_parameters,infer_parameters2, plotter,KF




# Write your package code here.
include("system_parameters.jl")
include("pulsars.jl")
include("GW.jl")
include("guessed_parameters.jl")
include("synthetic_data.jl")
include("model.jl")
include("priors.jl")
include("linear_kalman_filter.jl")
include("particle_swarm.jl")
include("particle_swarm_v2.jl")
include("plotting.jl")
include("run.jl")


end
