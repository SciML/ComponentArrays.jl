abstract type AbstractAxis{IdxMap} end

@inline indexmap(::AbstractAxis{IdxMap}) where {IdxMap} = IdxMap
@inline indexmap(ax::AbstractUnitRange) = ax
@inline indexmap(::Type{<:AbstractAxis{IdxMap}}) where {IdxMap} = IdxMap

# struct FlatAxis <: AbstractAxis{NamedTuple()} end

struct NullAxis <: AbstractAxis{nothing} end
const VarAxes = Tuple{Vararg{AbstractAxis}}

"""
    ax = Axis(nt::NamedTuple)

Gives named component access for `ComponentArray`s.

# Examples

```jldoctest
julia> using ComponentArrays

julia> ax = Axis((a = 1, b = ViewAxis(2:7, PartitionedAxis(2, (a = 1, b = 2))),
           c = ViewAxis(8:10, (a = 1, b = 2:3))));

julia> A = [100, 4, 1.3, 1, 1, 4.4, 0.4, 2, 1, 45];

julia> ca = ComponentArray(A, ax)
ComponentVector{Float64}(a = 100.0, b = ComponentVector{Float64, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Tuple{Axis{(a = 1, b = 2)}}}[(a = 4.0, b = 1.3), (a = 1.0, b = 1.0), (a = 4.4, b = 0.4)], c = (a = 2.0, b = [1.0, 45.0]))

julia> ca.a
100.0

julia> ca.b
3-element LazyArray{ComponentVector{Float64, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Tuple{Axis{(a = 1, b = 2)}}}}:
 ComponentVector{Float64,SubArray...}(a = 4.0, b = 1.3)
 ComponentVector{Float64,SubArray...}(a = 1.0, b = 1.0)
 ComponentVector{Float64,SubArray...}(a = 4.4, b = 0.4)

julia> ca.c
ComponentVector{Float64,SubArray...}(a = 2.0, b = [1.0, 45.0])

julia> ca.c.b
2-element view(::Vector{Float64}, 9:10) with eltype Float64:
  1.0
 45.0
```
"""
struct Axis{IdxMap} <: AbstractAxis{IdxMap} end
@inline Axis(IdxMap::NamedTuple) = Axis{IdxMap}()
Axis(; kwargs...) = Axis((; kwargs...))
function Axis(symbols::Union{AbstractVector{Symbol}, NTuple{N, Symbol}}) where {N}
    return Axis(NamedTuple{(symbols...,)}((eachindex(symbols)...,)))
end
Axis(symbols::Vararg{Symbol}) = Axis(symbols)

const FlatAxis = Axis{NamedTuple()}
const NullorFlatAxis = Union{NullAxis, FlatAxis}

"""
    sa = ShapedAxis(shape)

Preserves higher-dimensional array components in `ComponentArray`s (matrix components, for
example)
"""
struct ShapedAxis{Shape} <: AbstractAxis{nothing} end
@inline ShapedAxis(Shape) = ShapedAxis{Shape}()
# ShapedAxis(::Tuple{<:Int}) = FlatAxis()
Base.length(::ShapedAxis{Shape}) where {Shape} = prod(Shape)

struct Shaped1DAxis{Shape} <: AbstractAxis{nothing} end
ShapedAxis(shape::Tuple{<:Int}) = Shaped1DAxis{shape}()
Shaped1DAxis(shape::Tuple{<:Int}) = Shaped1DAxis{shape}()
Base.length(::Shaped1DAxis{Shape}) where {Shape} = only(Shape)

const Shape = ShapedAxis

unshape(ax) = ax
unshape(ax::ShapedAxis) = Axis(indexmap(ax))
unshape(ax::Shaped1DAxis) = Axis(indexmap(ax))

Base.size(::ShapedAxis{Shape}) where {Shape} = Shape
Base.size(::Shaped1DAxis{Shape}) where {Shape} = Shape

"""
    pa = PartitionedAxis(partition_size, index_map)

Axis for creating arrays of `ComponentArray`s
"""
struct PartitionedAxis{PartSz, IdxMap, Ax <: AbstractAxis{IdxMap}} <: AbstractAxis{IdxMap}
    ax::Ax

    function PartitionedAxis(PartSz, ax::AbstractAxis{IdxMap}) where {IdxMap}
        return new{PartSz, IdxMap, typeof(ax)}(ax)
    end
end
function PartitionedAxis{PartSz, IdxMap, Ax}() where {PartSz, IdxMap, Ax}
    PartitionedAxis(PartSz, Ax())
end
PartitionedAxis(PartSz, IdxMap) = PartitionedAxis(PartSz, Axis(IdxMap))

const Partition = PartitionedAxis

Base.size(::PartitionedAxis{PartSz, IdxMap}) where {PartSz, IdxMap} = PartSz
Base.size(::Type{PartitionedAxis{PartSz, IdxMap}}) where {PartSz, IdxMap} = PartSz

"""
    va = ViewAxis(parent_index, index_map)

Axis for creating arrays of `ComponentArray`s
"""
struct ViewAxis{Inds, IdxMap, Ax <: AbstractAxis{IdxMap}} <: AbstractAxis{IdxMap}
    ax::Ax
    function ViewAxis(Inds, ax::AbstractAxis{IdxMap}) where {IdxMap}
        return new{Inds, IdxMap, typeof(ax)}(ax)
    end
    ViewAxis(Inds, ::NullorFlatAxis) = Inds
end
# ViewAxis{Inds,IdxMap,Ax}() where {Inds,IdxMap,Ax} = PartitionedAxis(Inds, Ax())
ViewAxis{Inds, IdxMap, Ax}() where {Inds, IdxMap, Ax} = ViewAxis(Inds, Ax())
ViewAxis(Inds, IdxMap) = ViewAxis(Inds, Axis(IdxMap))
ViewAxis(Inds) = Inds

Base.length(ax::ViewAxis{Inds}) where {Inds} = length(Inds)
# Fix https://github.com/Deltares/Ribasim/issues/2028
function Base.getindex(::ViewAxis{Inds, IdxMap, <:ComponentArrays.Shaped1DAxis},
        idx::Integer) where {Inds, IdxMap}
    Inds[idx]
end
function Base.iterate(::ViewAxis{
        Inds, IdxMap, <:ComponentArrays.Shaped1DAxis}) where {Inds, IdxMap}
    iterate(Inds)
end
function Base.iterate(
        ::ViewAxis{
            Inds, IdxMap, <:ComponentArrays.Shaped1DAxis}, idx) where {Inds, IdxMap}
    iterate(Inds, idx)
end

const View = ViewAxis
const NullOrFlatView{Inds, IdxMap} = ViewAxis{Inds, IdxMap, <:NullorFlatAxis}

viewindex(::ViewAxis{Inds, IdxMap}) where {Inds, IdxMap} = Inds
viewindex(::Type{<:ViewAxis{Inds, IdxMap}}) where {Inds, IdxMap} = Inds
viewindex(i) = i

Axis(ax::AbstractAxis) = ax
Axis(ax::PartitionedAxis) = ax.ax
Axis(ax::ViewAxis) = ax.ax

# Get rid of this
Axis(::Number) = NullAxis()
Axis(::NamedTuple{()}) = FlatAxis()
Axis(x) = FlatAxis()

const NotShapedAxis = Union{Axis{IdxMap}, FlatAxis, NullAxis, Shaped1DAxis} where {IdxMap}
const NotPartitionedAxis = Union{
    Axis{IdxMap}, FlatAxis, NullAxis, ShapedAxis{Shape}, Shaped1DAxis} where {Shape, IdxMap}
const NotShapedOrPartitionedAxis = Union{
    Axis{IdxMap}, FlatAxis, Shaped1DAxis} where {IdxMap}

Base.merge(axs::Vararg{Axis}) = Axis(merge(indexmap.(axs)...))

# TODO: broken for FlatAxis
Base.firstindex(ax::AbstractAxis) = first(viewindex(first(indexmap(ax))))
Base.lastindex(ax::AbstractAxis) = last(viewindex(last(indexmap(ax))))
Base.length(ax::AbstractAxis) = lastindex(ax) - firstindex(ax) + 1
Base.length(ax::NullorFlatAxis) = error("NullorFlatAxis has no length")

Base.keys(ax::AbstractAxis) = keys(indexmap(ax))

reindex(i, offset) = i .+ offset
reindex(ax::FlatAxis, _) = ax
reindex(ax::Axis, offset) = Axis(map(x->reindex(x, offset), indexmap(ax)))
reindex(ax::ViewAxis, offset) = ViewAxis(viewindex(ax) .+ offset, indexmap(ax))
function reindex(ax::ViewAxis{OldInds, IdxMap, Ax},
        offset) where {OldInds, IdxMap, Ax <: Union{Shaped1DAxis, ShapedAxis}}
    NewInds = viewindex(ax) .+ offset
    return ViewAxis(NewInds, Ax())
end

# Get AbstractAxis index
@inline Base.getindex(::AbstractAxis, idx) = ComponentIndex(idx)
@inline Base.getindex(::AbstractAxis, idx::FlatIdx) = ComponentIndex(idx)
@inline Base.getindex(ax::AbstractAxis, ::Colon) = ComponentIndex(:, ax)
@inline Base.getindex(::AbstractAxis{IdxMap}, s::Symbol) where {IdxMap} = ComponentIndex(getproperty(IdxMap, s))
@inline Base.getindex(
    ::AbstractAxis{IdxMap}, ::Val{s}) where {
    IdxMap, s} = ComponentIndex(getproperty(IdxMap, s))
function Base.getindex(ax::AbstractAxis, syms::Union{
        NTuple{N, Symbol}, <:AbstractArray{Symbol}}) where {N}
    @assert allunique(syms) "Indexing symbols must all be unique. Got $syms"
    c_inds = getindex.((ax,), syms)
    inds = map(x->x.idx, c_inds)
    axs = map(x->x.ax, c_inds)
    last_index = 0
    new_axs = map(inds, axs) do i, ax
        first_index = last_index + 1
        last_index = last_index + length(i)
        _maybe_view_axis(first_index:last_index, ax)
    end
    new_ax = Axis(NamedTuple(syms .=> new_axs))
    return ComponentIndex(vcat(inds...), new_ax)
end

_maybe_view_axis(inds, ax::AbstractAxis) = ViewAxis(inds, ax)
_maybe_view_axis(inds, ::NullAxis) = inds[1]
_maybe_view_axis(inds, ax::Union{ShapedAxis, Shaped1DAxis}) = ViewAxis(inds, ax)

struct CombinedAxis{C, A} <: AbstractUnitRange{Int}
    component_axis::C
    array_axis::A
end

const CombinedOrRegularAxis = Union{Integer, AbstractUnitRange, CombinedAxis}

_component_axis(ax::CombinedAxis) = ax.component_axis
_component_axis(ax) = FlatAxis()

_array_axis(ax::CombinedAxis) = ax.array_axis
_array_axis(ax) = ax
_array_axis(ax::Int) = Shaped1DAxis((ax,))

Base.first(ax::CombinedAxis) = first(_array_axis(ax))

Base.last(ax::CombinedAxis) = last(_array_axis(ax))

Base.firstindex(ax::CombinedAxis) = firstindex(_array_axis(ax))

Base.lastindex(ax::CombinedAxis) = lastindex(_array_axis(ax))

Base.getindex(ax::CombinedAxis, i::Integer) = _array_axis(ax)[i]
Base.getindex(ax::CombinedAxis, i::AbstractArray) = _array_axis(ax)[i]

Base.length(ax::CombinedAxis) = lastindex(ax) - firstindex(ax) + 1

function Base.CartesianIndices(ax::Tuple{CombinedAxis, Vararg{CombinedAxis}})
    CartesianIndices(_array_axis.(ax))
end
