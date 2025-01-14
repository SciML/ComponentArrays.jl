module ComponentArraysTrackerExt

using ArrayInterface: ArrayInterface
using ComponentArrays, Tracker

function Tracker.param(ca::ComponentArray)
    x = getdata(ca)
    length(x) == 0 && return ComponentArray(Tracker.param(Float32[]), getaxes(ca))
    return ComponentArray(Tracker.param(x), getaxes(ca))
end

Tracker.extract_grad!(ca::ComponentArray) = Tracker.extract_grad!(getdata(ca))

Tracker.data(ca::ComponentArray) = ComponentArray(Tracker.data(getdata(ca)), getaxes(ca))

function Base.materialize(bc::Base.Broadcast.Broadcasted{Tracker.TrackedStyle,Nothing,
    typeof(zero),<:Tuple{<:ComponentVector}})
    ca = first(bc.args)
    return ComponentArray(zero.(getdata(ca)), getaxes(ca))
end

function Base.getindex(g::Tracker.Grads, x::ComponentArray)
    Tracker.istracked(getdata(x)) || error("Object not tracked: $x")
    id = Tracker.tracker(getdata(x))
    try
        return g[id]
    catch err
        # if length(keys(g.grads)) == 2
        #     dx = getdata(x)
        #     idx = findfirst(gg -> size(gg) == size(dx), collect(values(g.grads)))
        #     @assert idx !== nothing
        #     return collect(values(g.grads))[idx]
        # end
        return zero.(getdata(x))
    end
    # return g[id]
end

# For TrackedArrays ignore Base.maybeview
## Tracker with views doesn't work quite well
@inline function Base.getproperty(x::ComponentVector{T,<:TrackedArray},
    s::Symbol) where {T}
    return getproperty(x, Val(s))
end

@inline function Base.getproperty(x::ComponentVector{T,<:TrackedArray}, v::Val) where {T}
    return ComponentArrays._getindex(Base.getindex, x, v)
end

function ArrayInterface.restructure(x::ComponentVector,
    y::ComponentVector{T,<:TrackedArray}) where {T}
    getaxes(x) == getaxes(y) || error("Axes must match")
    return y
end

end
