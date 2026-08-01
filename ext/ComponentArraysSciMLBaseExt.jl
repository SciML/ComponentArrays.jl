# SciMLBase + SymbolicIndexingInterface integration for ComponentArrays
module ComponentArraysSciMLBaseExt

using ComponentArrays
using SciMLBase: SciMLBase
using SymbolicIndexingInterface: SymbolicIndexingInterface as SII

"""
Index provider that exposes top-level `ComponentArray` keys as symbolic variables
(and parameters, when `p` is also a `ComponentArray`).

Used when an `AbstractODEProblem` has `ComponentArray` state and the problem
function does not already carry a symbolic system (`sys`).
"""
struct ComponentArrayIndexProvider{U, P}
    u0::U
    p::P
end

function SII.is_variable(sc::ComponentArrayIndexProvider, sym)
    return SII.symbolic_type(sym) != SII.NotSymbolic() && sym in keys(sc.u0)
end

function SII.variable_index(sc::ComponentArrayIndexProvider, sym)
    return SII.is_variable(sc, sym) ? sym : nothing
end

function SII.variable_symbols(sc::ComponentArrayIndexProvider)
    return collect(Symbol, keys(sc.u0))
end
SII.variable_symbols(sc::ComponentArrayIndexProvider, _) = SII.variable_symbols(sc)

function SII.is_parameter(sc::ComponentArrayIndexProvider, sym)
    sc.p isa ComponentArray || return false
    return SII.symbolic_type(sym) != SII.NotSymbolic() && sym in keys(sc.p)
end

function SII.parameter_index(sc::ComponentArrayIndexProvider, sym)
    return SII.is_parameter(sc, sym) ? sym : nothing
end

function SII.parameter_symbols(sc::ComponentArrayIndexProvider)
    sc.p isa ComponentArray || return Symbol[]
    return collect(Symbol, keys(sc.p))
end

SII.is_independent_variable(::ComponentArrayIndexProvider, sym) = isequal(sym, :t)
SII.independent_variable_symbols(::ComponentArrayIndexProvider) = [:t]
SII.is_time_dependent(::ComponentArrayIndexProvider) = true
SII.constant_structure(::ComponentArrayIndexProvider) = true
SII.is_observed(::ComponentArrayIndexProvider, _) = false
SII.is_markovian(::ComponentArrayIndexProvider) = true
SII.all_variable_symbols(sc::ComponentArrayIndexProvider) = SII.variable_symbols(sc)
function SII.all_symbols(sc::ComponentArrayIndexProvider)
    return vcat(
        SII.variable_symbols(sc),
        SII.parameter_symbols(sc),
        SII.independent_variable_symbols(sc),
    )
end

"""
When `u0` is a `ComponentArray` and the ODE function has no symbolic `sys`,
expose top-level component names through SymbolicIndexingInterface so that
`sol[:x]`, `sol(t; idxs=:x)`, and `sol.ps[:a]` work.
"""
function SII.symbolic_container(
        prob::SciMLBase.AbstractODEProblem{uType},
    ) where {uType <: ComponentArray}
    f = prob.f
    if SciMLBase.has_sys(f)
        return f
    end
    return ComponentArrayIndexProvider(prob.u0, prob.p)
end

# Plotting labels for solutions of ComponentArray ODEs
function SciMLBase.getsyms(
        sol::SciMLBase.AbstractODESolution{
            T, N, C,
        },
    ) where {T, N, C <: AbstractVector{<:ComponentArray}}
    if SciMLBase.has_syms(sol.prob.f)
        return sol.prob.f.syms
    else
        return Symbol.(labels(sol.u[1]))
    end
end

end
