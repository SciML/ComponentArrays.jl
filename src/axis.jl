"""
    AbstractAxis{IdxMap}

Abstract supertype for axis metadata used by `ComponentArray` to map component
names and shaped views onto positions in the wrapped array.

# Type Parameters

  - `IdxMap`: A static `NamedTuple` mapping component names to component indices or
    nested axis metadata. It is stored as a type parameter so generic indexing can
    resolve the map without per-instance storage.

# Interface

Subtypes represent static component metadata. A subtype must use an `IdxMap` whose keys
are the component names accepted by the axis and whose values are valid component
indices, ranges, nested named tuples, or axis metadata supported by `ComponentArray`.
The map must describe the same component layout for every instance of the subtype, and
the subtype must be constructible without storing a second, runtime copy of the map.

The generic interface is derived from `IdxMap`; a subtype normally does not implement
any methods itself:

  - `keys(axis)` returns the component names in `IdxMap` order.
  - `axis[name]` and `axis[Val(name)]` return a `ComponentIndex` for one named component.
  - `axis[names]`, where `names` is a tuple or array of symbols, returns one
    `ComponentIndex` whose indices are concatenated in the requested order.
  - `firstindex(axis)` and `lastindex(axis)` describe the first and last flattened
    positions represented by the map.

When an axis is passed to [`ComponentArray`](@ref), its map must describe exactly the
component positions in the supplied data, including any nested or shaped metadata.
Define a custom subtype only when a distinct, statically known axis representation is
required. Use [`Axis`](@ref) for ordinary named component layouts.

# Examples

```jldoctest
julia> using ComponentArrays

julia> struct TwoComponentAxis <: AbstractAxis{(left = 1, right = 2)} end

julia> ax = TwoComponentAxis();

julia> keys(ax)
(:left, :right)

```
"""
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

julia> ax = Axis(
           (
               a = 1, b = ViewAxis(2:7, PartitionedAxis(2, (a = 1, b = 2))),
               c = ViewAxis(8:10, (a = 1, b = 2:3)),
           )
       );

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

"""
    FlatAxis()

Axis marker for an unnamed, flat dimension of a `ComponentArray`.

# Examples

```jldoctest
julia> using ComponentArrays

julia> x = ComponentArray(reshape(1:4, 2, 2), Axis(row = 1:2), FlatAxis());

julia> getaxes(x)
(Axis(row = 1:2,), FlatAxis())
```
"""
const FlatAxis = Axis{NamedTuple()}
const NullorFlatAxis = Union{NullAxis, FlatAxis}

"""
    ShapedAxis(shape::Tuple{Vararg{Integer}})

Axis metadata that preserves the shape of a multidimensional component stored in a flat
`ComponentArray` data buffer.

# Arguments

  - `shape`: The dimensions of the component. A one-dimensional `shape` produces a
    [`Shaped1DAxis`](@ref) instead.

# Examples

```jldoctest
julia> using ComponentArrays

julia> size(ShapedAxis((2, 3)))
(2, 3)
```
"""
struct ShapedAxis{Shape} <: AbstractAxis{nothing} end
@inline ShapedAxis(Shape) = ShapedAxis{Shape}()
# ShapedAxis(::Tuple{<:Int}) = FlatAxis()
Base.length(::ShapedAxis{Shape}) where {Shape} = prod(Shape)

"""
    Shaped1DAxis(shape::Tuple{<:Integer})

Axis marker for a one-dimensional array component. `ShapedAxis((n,))` returns a
`Shaped1DAxis` so vector-valued components keep their one-dimensional shape.

# Examples

```jldoctest
julia> using ComponentArrays

julia> ax = Shaped1DAxis((3,));

julia> size(ax)
(3,)
```
"""
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
    PartitionedAxis(partition_size, index_map)

Axis metadata for a homogeneous array of component layouts. Constructing a
`ComponentArray` with a `PartitionedAxis` produces a lazy array whose entries are
`ComponentArray`s sharing the same component map.

# Arguments

  - `partition_size`: Number of flat data elements in each component layout.
  - `index_map`: A `NamedTuple` or [`AbstractAxis`](@ref) describing one layout.

# Examples

```jldoctest
julia> using ComponentArrays

julia> axis = PartitionedAxis(2, (x = 1, y = 2));

julia> size(axis)
2
```
"""
struct PartitionedAxis{PartSz, IdxMap, Ax <: AbstractAxis{IdxMap}} <: AbstractAxis{IdxMap}
    ax::Ax

    function PartitionedAxis(PartSz, ax::AbstractAxis{IdxMap}) where {IdxMap}
        return new{PartSz, IdxMap, typeof(ax)}(ax)
    end
end
function PartitionedAxis{PartSz, IdxMap, Ax}() where {PartSz, IdxMap, Ax}
    return PartitionedAxis(PartSz, Ax())
end
PartitionedAxis(PartSz, IdxMap) = PartitionedAxis(PartSz, Axis(IdxMap))

const Partition = PartitionedAxis

Base.size(::PartitionedAxis{PartSz, IdxMap}) where {PartSz, IdxMap} = PartSz
Base.size(::Type{PartitionedAxis{PartSz, IdxMap}}) where {PartSz, IdxMap} = PartSz

"""
    ViewAxis(parent_index, index_map)

Axis metadata that maps a component layout onto `parent_index` in its parent array.
`ViewAxis` preserves nested component names while recording the parent positions used to
retrieve the component. For flat and null axes it simplifies to the bare index.

# Arguments

  - `parent_index`: Indices of the component in the parent array.
  - `index_map`: A `NamedTuple` or [`AbstractAxis`](@ref) describing the component layout.

# Examples

```jldoctest
julia> using ComponentArrays

julia> axis = ViewAxis(2:3, (x = 1, y = 2));

julia> keys(axis)
(:x, :y)
```
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
function Base.getindex(
        ::ViewAxis{Inds, IdxMap, <:ComponentArrays.Shaped1DAxis},
        idx::Integer
    ) where {Inds, IdxMap}
    return Inds[idx]
end
function Base.iterate(
        ::ViewAxis{
            Inds, IdxMap, <:ComponentArrays.Shaped1DAxis,
        }
    ) where {Inds, IdxMap}
    return iterate(Inds)
end
function Base.iterate(
        ::ViewAxis{
            Inds, IdxMap, <:ComponentArrays.Shaped1DAxis,
        }, idx
    ) where {Inds, IdxMap}
    return iterate(Inds, idx)
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
    Axis{IdxMap}, FlatAxis, NullAxis, ShapedAxis{Shape}, Shaped1DAxis,
} where {Shape, IdxMap}
const NotShapedOrPartitionedAxis = Union{
    Axis{IdxMap}, FlatAxis, Shaped1DAxis,
} where {IdxMap}

Base.merge(axs::Vararg{Axis}) = Axis(merge(indexmap.(axs)...))

Base.firstindex(ax::AbstractAxis) = first(viewindex(first(indexmap(ax))))
Base.lastindex(ax::AbstractAxis) = last(viewindex(last(indexmap(ax))))

Base.keys(ax::AbstractAxis) = keys(indexmap(ax))

reindex(i, offset) = i .+ offset
reindex(ax::FlatAxis, _) = ax
reindex(ax::Axis, offset) = Axis(map(x -> reindex(x, offset), indexmap(ax)))
reindex(ax::ViewAxis, offset) = ViewAxis(viewindex(ax) .+ offset, indexmap(ax))
function reindex(
        ax::ViewAxis{OldInds, IdxMap, Ax},
        offset
    ) where {OldInds, IdxMap, Ax <: Union{Shaped1DAxis, ShapedAxis}}
    NewInds = viewindex(ax) .+ offset
    return ViewAxis(NewInds, Ax())
end

# Get AbstractAxis index
@inline Base.getindex(::AbstractAxis, idx) = ComponentIndex(idx)
@inline Base.getindex(::AbstractAxis, idx::FlatIdx) = ComponentIndex(idx)
@inline Base.getindex(ax::AbstractAxis, ::Colon) = ComponentIndex(:, ax)
@inline Base.getindex(::AbstractAxis{IdxMap}, s::Symbol) where {IdxMap} = ComponentIndex(getproperty(IdxMap, s))
@inline Base.getindex(
    ::AbstractAxis{IdxMap}, ::Val{s}
) where {
    IdxMap, s,
} = ComponentIndex(getproperty(IdxMap, s))
function Base.getindex(
        ax::AbstractAxis, syms::Union{
            NTuple{N, Symbol}, <:AbstractArray{Symbol},
        }
    ) where {N}
    @assert allunique(syms) "Indexing symbols must all be unique. Got $syms"
    c_inds = getindex.((ax,), syms)
    inds = map(x -> x.idx, c_inds)
    axs = map(x -> x.ax, c_inds)
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
    return CartesianIndices(_array_axis.(ax))
end
