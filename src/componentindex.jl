struct ComponentIndex{Idx, Ax <: AbstractAxis}
    idx::Idx
    ax::Ax
end
ComponentIndex(idx) = ComponentIndex(idx, FlatAxis())
ComponentIndex(idx::CartesianIndex) = ComponentIndex(idx, ShapedAxis((1,)))
ComponentIndex(idx::AbstractArray{<:Integer}) = ComponentIndex(idx, ShapedAxis(size(idx)))
ComponentIndex(idx::Int) = ComponentIndex(idx, NullAxis())
function ComponentIndex(vax::ViewAxis{Inds, IdxMap, Ax}) where {Inds, IdxMap, Ax}
    return ComponentIndex(Inds, vax.ax)
end

const FlatComponentIndex{Idx} = ComponentIndex{Idx, FlatAxis}
const NullComponentIndex{Idx} = ComponentIndex{Idx, NullAxis}

function Base.:(==)(ci1::ComponentIndex, ci2::ComponentIndex)
    return ci1.idx == ci2.idx && ci1.ax == ci2.ax
end

Base.length(ci::ComponentIndex) = length(ci.idx)

"""
    KeepIndex(idx)

Tag an index of a `ComponentArray` to retain it's `Axis` through indexing
"""
struct KeepIndex{Idx} end
KeepIndex(idx) = KeepIndex{idx}()
KeepIndex(idx::Integer) = KeepIndex(idx:idx)

Base.getindex(ax::AbstractAxis, i::KeepIndex{Idx}) where {Idx} = _getindex_keep(ax, Idx)

_getindex_keep(ax::AbstractAxis, ::Colon) = ComponentIndex(:, ax)
function _getindex_keep(ax::AbstractAxis, idx::AbstractRange)
    idx_map = indexmap(ax)
    keeps = (
        s => x for (s, x) in pairs(idx_map) if first(viewindex(x)) in idx &&
            last(viewindex(x)) in idx
    )
    keeps = NamedTuple{Tuple(first.(keeps))}(Tuple(last.(keeps)))
    new_ax = reindex(Axis(keeps), -first(idx) + 1)
    return ComponentIndex(idx, new_ax)
end
function _getindex_keep(ax::AbstractAxis, sym::Symbol)
    ci = ax[sym]
    idx = ci.idx
    if idx isa Integer
        idx = idx:idx
    end
    if ci.ax isa NullAxis || ci.ax isa FlatAxis
        new_ax = Axis(NamedTuple{(sym,)}((ci.idx,)))
    else
        new_ax = Axis(NamedTuple{(sym,)}((ViewAxis(idx, ci.ax),)))
    end
    new_ax = reindex(new_ax, -first(idx) + 1)
    return ComponentIndex(idx, new_ax)
end
