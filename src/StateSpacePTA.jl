module StateSpacePTA

#Imports 
import Parameters: @with_kw, @unpack
#using LinearAlgebra,DifferentialEquations, Noise,Plots,Statistics,DelimitedFiles, JLD, Random,Logging

using CSV,DataFrames, LinearAlgebra, Random, Noise, Plots, Statistics,BAT, IntervalSets,Distributions,DifferentialEquations
#DataFrames, CSV

import StatsBase: Histogram #just for testing, can remove later

import Suppressor: @suppress_err





#Exports
export KalmanFilter, plotter




# Write your package code here.
include("system_parameters.jl")
include("pulsars.jl")
include("GW.jl")
include("guessed_parameters.jl")
include("synthetic_data.jl")
include("model.jl")
#include("unscented_kalman_filter.jl")
#include("extended_kalman_filter.jl")
include("bayesian_inference.jl")
include("linear_kalman_filter.jl")

include("plotting.jl")
include("run.jl")


end
