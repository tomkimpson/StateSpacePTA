module StateSpacePTA

#Imports 
import Parameters: @with_kw, @unpack
#using LinearAlgebra,DifferentialEquations, Noise,Plots,Statistics,DelimitedFiles, JLD, Random,Logging

using CSV,DataFrames, LinearAlgebra, Random, Noise, Plots, Statistics, IntervalSets,Distributions,DifferentialEquations
#DataFrames, CSV

import StatsBase: Histogram #just for testing, can remove later

import Suppressor: @suppress_err


using TensorOperations

#using Optim

using ValueShapes

# #extras for NEstedSamplers
using NestedSamplers,AbstractMCMC
using StatsFuns: logaddexp
using StatsBase: sample, Weights


AbstractMCMC.setprogress!(true)
Random.seed!(8452)

#Exports
export setup, KF, run_all, plotter,parameter_estimation #KalmanFilter,infer_parameters,infer_parameters2, plotter,KF




# Write your package code here.
include("system_parameters.jl")
include("pulsars.jl")
include("GW.jl")
include("guessed_parameters.jl")
include("synthetic_data.jl")
include("model.jl")
include("priors.jl")
include("linear_kalman_filter.jl")
#include("nested_samplers.jl")
include("plotting.jl")
include("run.jl")


end
