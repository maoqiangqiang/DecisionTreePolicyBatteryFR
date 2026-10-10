from juliacall import Main as jl



jl.seval("using Pkg")
jl.seval('Pkg.add("DifferentialEquations")')
jl.seval('Pkg.add("Sundials")')
jl.seval('Pkg.add("Interpolations")')
jl.seval('Pkg.add("Distributions")')
jl.seval('Pkg.add("JuMP")')
jl.seval('Pkg.add("Ipopt")')
jl.seval('Pkg.add("Statistics")')
jl.seval('Pkg.add("DelimitedFiles")')
jl.seval('Pkg.add("LinearAlgebra")')
jl.seval('Pkg.add("StatsBase")')
jl.seval('Pkg.add("Random")')
jl.seval('Pkg.add("JLD")')
jl.seval('Pkg.add("BSON")')
jl.seval('Pkg.add("ZMQ")')
jl.seval('Pkg.add("JSON")')
jl.seval('Pkg.add("GLPK")')
jl.seval('Pkg.add("CSV")')
jl.seval('Pkg.add("DataFrames")')


