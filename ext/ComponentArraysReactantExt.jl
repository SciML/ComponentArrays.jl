module ComponentArraysReactantExt

using ArrayInterface: ArrayInterface
using ComponentArrays, Reactant

const TracedComponentVector{T} = ComponentVector{
    Reactant.TracedRNumber{T}, <:Reactant.TracedRArray{T},
} where {T}

# Reactant is good at memory management but not great at handling wrapped types. So we avoid
# wrapping types into SubArrays and let Reactant optimize out intermediate allocations.

@inline function Base.getproperty(x::TracedComponentVector{T}, s::Symbol) where {T}
    return getproperty(x, Val(s))
end

@inline function Base.getproperty(x::TracedComponentVector{T}, v::Val) where {T}
    return ComponentArrays._getindex(Base.getindex, x, v)
end

function ArrayInterface.restructure(x::ComponentVector, y::TracedComponentVector)
    getaxes(x) == getaxes(y) || error("Axes must match")
    return y
end

end
