module RLEnvironments
__precompile__()

using Distributions: Uniform
import Distributions: sample

abstract type Environment{S,A,R<:Real} end
abstract type DiscEnv{S,A,R} <: Environment{S,A,R} end
abstract type ContEnv{S,A,R} <: Environment{S,A,R} end
abstract type DiffEnv{S,A,R} <: Environment{S,A,R} end

struct ActionSpace{A}
    actions::Vector{Tuple{A,A}}
end
Base.length(a::ActionSpace) = length(a.actions)


function sample(a::ActionSpace{T}) where T <: AbstractFloat
    return map(x -> T(rand(Uniform(x[1], x[2]))), a.actions)
end

struct ObservationSpace{S}
    n::Int64
end
Base.length(o::ObservationSpace) = o.n



mutable struct FR_battery{S,A,R}
    current::Vector{S}
    observationspace::ObservationSpace{S}
    actionspace::ActionSpace{A}
    state::Vector{S}

    function FR_battery{T}(state_dim, inner_state_dim) where T <: AbstractFloat
        observationspace = ObservationSpace{T}(state_dim)
        actionspace = ActionSpace([(-one(T), one(T)),(-one(T),one(T))])
        new{T, T, T}(zeros(T,state_dim), observationspace, actionspace, zeros(T, inner_state_dim))
    end
end


export FR_battery


end
