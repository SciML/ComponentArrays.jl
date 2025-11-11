"""
    x = ComponentArray(nt::NamedTuple)
    x = ComponentArray(;kwargs...)
    x = ComponentArray(data::AbstractVector, ax)
    x = ComponentArray{T}(args...; kwargs...) where T

Array type that can be accessed like an arbitrary nested mutable struct.

# Examples

```jldoctest
julia> using ComponentArrays

julia> x = ComponentArray(a = 1, b = [2, 1, 4], c = (a = 2, b = [1, 2]))
ComponentVector{Int64}(a = 1, b = [2, 1, 4], c = (a = 2, b = [1, 2]))

julia> x.c.a = 400;
       x
ComponentVector{Int64}(a = 1, b = [2, 1, 4], c = (a = 400, b = [1, 2]))

julia> x[5]
400

julia> collect(x)
7-element Vector{Int64}:
   1
   2
   1
   4
 400
   1
   2
```
"""
struct ComponentArray{
    T, N, A <: AbstractArray{T, N}, Axes <: Tuple{Vararg{AbstractAxis}}} <: DenseArray{T, N}
    data::A
    axes::Axes
end

# Entry from type (used for broadcasting)
ComponentArray{Axes}(data) where {Axes} = ComponentArray(data, getaxes(Axes)...)
function ComponentArray(::UndefInitializer, ax::Axes) where {Axes <: Tuple}
    ComponentArray(similar(Array{Float64}, last_index.(ax)), ax...)
end
function ComponentArray{A}(::UndefInitializer, ax::Axes) where {
        A <: AbstractArray, Axes <: Tuple}
    ComponentArray(similar(A, last_index.(ax)), ax...)
end
function ComponentArray{T}(::UndefInitializer, ax::Axes) where {T, Axes <: Tuple}
    ComponentArray(similar(Array{T}, last_index.(ax)), ax...)
end

# Entry from data array and AbstractAxis types dispatches to correct shapes and partitions
# then packs up axes into a tuple for inner constructor
# ComponentArray(data, ::FlatAxis...) = data
ComponentArray(data, ::Union{FlatAxis, Shaped1DAxis}...) = data
ComponentArray(data, ax::NotShapedOrPartitionedAxis...) = ComponentArray(data, ax)
function ComponentArray(data, ax::NotPartitionedAxis...)
    ComponentArray(maybe_reshape(data, ax...), unshape.(ax)...)
end
function ComponentArray(data, ax::AbstractAxis...)
    part_axs = filter_by_type(PartitionedAxis, ax...)
    part_data = partition(data, size.(part_axs)...)
    axs = Axis.(ax)
    # return [ComponentArray(x, axs...) for x in part_data]
    return LazyArray(ComponentArray(x, axs...) for x in part_data)
end

# This code-path is hit for empty NamedTuples. We force a Float32 eltype here if
# length(data) == 0
function ComponentArray(data::Vector{Any}, axes::Tuple{FlatAxis})
    length(data) == 0 && return ComponentArray(Float32[], axes)
    return ComponentArray{Any, 1, typeof(data), typeof(axes)}(data, axes)
end

function Adapt.adapt_structure(to, x::ComponentArray)
    data = Adapt.adapt(to, getdata(x))
    return ComponentArray(data, getaxes(x))
end

function Adapt.adapt_storage(::Type{ComponentArray{T, N, A, Ax}},
        xs::AT) where {T, N, A, Ax, AT <: AbstractArray}
    Adapt.adapt_storage(A, xs)
end

Adapt.parent_type(::Type{ComponentArray{T, N, A, Ax}}) where {T, N, A, Ax} = A

# Entry from NamedTuple, Dict, or kwargs
ComponentArray{T}(nt::NamedTuple) where {T} = ComponentArray(make_carray_args(T, nt)...)
ComponentArray{T}(::NamedTuple{(), Tuple{}}) where {T} = ComponentArray(T[], (FlatAxis(),))
function ComponentArray(nt::Union{NamedTuple, AbstractDict})
    ComponentArray(make_carray_args(nt)...)
end
ComponentArray(::NamedTuple{(), Tuple{}}) = ComponentArray(Any[], (FlatAxis(),))
ComponentArray{T}(; kwargs...) where {T} = ComponentArray{T}((; kwargs...))
ComponentArray(; kwargs...) = ComponentArray((; kwargs...))

ComponentArray(x::ComponentArray) = x
ComponentArray{T}(x::ComponentArray) where {T} = T.(x)
function (CA::Type{<:ComponentArray{T, N, A, Ax}})(x::ComponentArray) where {T, N, A, Ax}
    ComponentArray(T.(getdata(x)), getaxes(x))
end

function fill_componentarray_ka! end # defined in extensions

## Some aliases
"""
    x = ComponentVector(nt::NamedTuple)
    x = ComponentVector(;kwargs...)
    x = ComponentVector(data::AbstractVector, ax)
    x = ComponentVector{T}(args...; kwargs...) where T

A `ComponentVector` is an alias for a one-dimensional `ComponentArray`.
"""
const ComponentVector{T, A, Axes} = ComponentArray{T, 1, A, Axes}
ComponentVector(nt) = ComponentArray(nt)
ComponentVector{T}(nt) where {T} = ComponentArray{T}(nt)
ComponentVector(; kwargs...) = ComponentArray(; kwargs...)
ComponentVector{T}(; kwargs...) where {T} = ComponentArray{T}(; kwargs...)
ComponentVector{T}(::UndefInitializer, ax) where {T} = ComponentArray{T}(undef, ax)
ComponentVector(data::AbstractVector, ax) = ComponentArray(data, ax)
function ComponentVector(data::AbstractArray, ax)
    throw(DimensionMismatch("A `ComponentVector` must be initialized with a 1-dimensional array. This array is $(ndims(data))-dimensional."))
end

function ConstructionBase.setproperties(x::ComponentVector, patch::NamedTuple)
    ComponentVector(x; patch...)
end

# Add new fields to component Vector
function ComponentArray(x::ComponentVector; kwargs...)
    return foldl((x1, kwarg) -> _maybe_add_field(x1, kwarg), (kwargs...,); init = x)
end
ComponentVector(x::ComponentVector; kwargs...) = ComponentArray(x; kwargs...)

ComponentVector{T}(x::ComponentVector) where {T} = T.(x)

"""
    x = ComponentMatrix(data::AbstractMatrix, ax...)
    x = ComponentMatrix{T}(data::AbstractMatrix, ax...) where T

A `ComponentMatrix` is an alias for a two-dimensional `ComponentArray`.
"""
const ComponentMatrix{T, A, Axes} = ComponentArray{T, 2, A, Axes}
ComponentMatrix{T}(::UndefInitializer, ax...) where {T} = ComponentArray{T}(undef, ax...)
ComponentMatrix(data::AbstractMatrix, ax...) = ComponentArray(data, ax...)
function ComponentMatrix(data::AbstractArray, ax...)
    throw(DimensionMismatch("A `ComponentMatrix` must be initialized with a 2-dimensional array. This array is $(ndims(data))-dimensional."))
end

ComponentMatrix(x::ComponentMatrix) = x
ComponentMatrix{T}(x::ComponentMatrix) where {T} = T.(x)

ComponentMatrix() = ComponentMatrix(Array{Any}(undef, 0, 0), (FlatAxis(), FlatAxis()))
function ComponentMatrix{T}() where {T}
    ComponentMatrix(Array{T}(undef, 0, 0), (FlatAxis(), FlatAxis()))
end

const CArray = ComponentArray
const CVector = ComponentVector
const CMatrix = ComponentMatrix

const AdjOrTrans{T, A} = Union{Adjoint{T, A}, Transpose{T, A}}
const AdjOrTransComponentArray{
    T, A} = Union{Adjoint{T, A}, Transpose{T, A}} where {A <: ComponentArray}
const AdjOrTransComponentVector{T} = Union{
    Adjoint{T, A}, Transpose{T, A}} where {A <: ComponentVector}
const AdjOrTransComponentMatrix{T} = Union{
    Adjoint{T, A}, Transpose{T, A}} where {A <: ComponentMatrix}

const ComponentVecOrMat{T} = Union{ComponentVector{T}, ComponentMatrix{T}} where {T}
const AdjOrTransComponentVecOrMat{T} = AdjOrTrans{T, <:ComponentVecOrMat} where {T}
const AbstractComponentArray{T} = Union{
    ComponentArray{T}, AdjOrTransComponentArray{T}} where {T}
const AbstractComponentVecOrMat{T} = Union{
    ComponentVecOrMat{T}, AdjOrTransComponentVecOrMat{T}} where {T}
const AbstractComponentVector{T} = Union{
    ComponentVector{T}, AdjOrTransComponentVector{T}} where {T}
const AbstractComponentMatrix{T} = Union{
    ComponentMatrix{T}, AdjOrTransComponentMatrix{T}} where {T}

## Constructor helpers
allocate_numeric_container(x) = allocate_numeric_container(recursive_eltype(x))
allocate_numeric_container(::Type{T}) where {T <: Number} = T[]
allocate_numeric_container(::Type) = []

# For making ComponentArrays from named tuples
make_carray_args(::NamedTuple{(), Tuple{}}) = (Any[], FlatAxis())
make_carray_args(::Type{T}, ::NamedTuple{(), Tuple{}}) where {T} = (T[], FlatAxis())
function make_carray_args(nt)
    data, ax = make_carray_args(Vector, nt)
    data = length(data)==1 ? [data[1]] : map(identity, data)
    return (data, ax)
end
make_carray_args(::Type{T}, nt) where {T} = make_carray_args(Vector{T}, nt)
function make_carray_args(A::Type{<:AbstractArray}, nt)
    init = allocate_numeric_container(nt)
    data, idx = make_idx(init, nt, 0)
    return (A(data), Axis(idx))
end

# Builds up data vector and returns appropriate AbstractAxis type for each input type
function make_idx(data, nt::Union{NamedTuple, AbstractDict}, last_val)
    len = recursive_length(nt)
    lv = Ref(0) # workaround for https://github.com/JuliaLang/julia/issues/15276
    kvs = (; (
        k => begin
            inds = make_idx(data, v, lv[])[2]
            lv[] = last_index(inds)
            inds
        end
    for (k, v) in pairs(nt)
    )...)
    return (data, ViewAxis(last_index(last_val) .+ (1:len), kvs))
end
function make_idx(data, nt::NamedTuple{(), Tuple{}}, last_val)
    out = last_index(last_val) .+ (1:length(nt))
    return (data, ViewAxis(out, ShapedAxis((length(nt),))))
end
function make_idx(data, pair::Pair, last_val)
    data, ax = make_idx(data, pair.second, last_val)
    len = recursive_length(data)
    return (data, ViewAxis(last_val:(last_val + len - 1), Axis(pair.second)))
end
make_idx(data, x, last_val) = (
    push!(data, x),
    ViewAxis(last_index(last_val) + 1)
)
function make_idx(data, x::ComponentVector, last_val)
    (
        append!(data, x),
        ViewAxis(
            last_index(last_val) .+ (1:length(x)),
            getaxes(x)[1]
        )
    )
end
function make_idx(data, x::AbstractArray, last_val)
    append!(data, x)
    out = last_index(last_val) .+ (1:length(x))
    return (data, ViewAxis(out, ShapedAxis(size(x))))
end
function make_idx(data, x::A, last_val) where {A <: AbstractArray{<:Union{
        NamedTuple, AbstractArray}}}
    len = recursive_length(x)
    elem_len = len ÷ length(x)
    if eltype(x) |> isconcretetype && all(elem -> recursive_length(elem) == elem_len, x)
        out = ()
        for elem in x
            _, out = make_idx(data, elem, last_val)
        end
        return (
            data,
            ViewAxis(
                last_index(last_val) .+ (1:len),
                PartitionedAxis(
                    elem_len,
                    indexmap(out)
                )
            )
        )
    else
        error("Only homogeneous arrays are allowed.")
    end
end
# function make_idx(data, x::A, last_val) where {A<:AbstractArray{<:AbstractArray}}
#     error("ComponentArrays cannot currently contain arrays of arrays as elements. This one contains: \n $x\n")
# end

#TODO: Make all internal function names start with underscores
function _maybe_add_field(x, pair)
    haskey(x, pair.first) ? _update_field(x, pair) : _add_field(x, pair)
end
function _add_field(x, pair)
    data = copy(getdata(x))
    new_data, new_ax = make_idx(data, pair.second, length(data))
    new_ax = Axis(NamedTuple{tuple(pair.first)}(tuple(new_ax)))
    new_ax = merge(getaxes(x)[1], new_ax)
    return ComponentArray(new_data, new_ax)
end
function _update_field(x, pair)
    x_copy = copy(x)
    x_copy[pair.first] = pair.second
    return x_copy
end

# Reshape ComponentArrays with ShapedAxis axes
maybe_reshape(data, ::NotShapedOrPartitionedAxis...) = data
function maybe_reshape(data, axs::AbstractAxis...)
    shapes = filter_by_type(Union{ShapedAxis, Shaped1DAxis}, axs...) .|> size
    shapes = reduce((tup, s) -> (tup..., s...), shapes)
    return reshape(data, shapes)
end

function Base.reshape(A::AbstractArray, axs::Tuple{CombinedAxis, Vararg{CombinedAxis}})
    reshape(A, _array_axis.(axs))
end

# Recurse through nested ViewAxis types to find the last index
last_index(x) = last(x)
last_index(x::ViewAxis) = last_index(viewindex(x))
last_index(x::AbstractAxis) = last_index(last(indexmap(x)))
function last_index(f::FlatAxis)
    nt = indexmap(f)
    length(nt) == 0 && return 0
    return last_index(last(nt))
end

# length information is in Axis, use it to make SVector creation type stable
@inline _hasNullOrFlatAxis(ca) = any(map(ax -> ax isa NullorFlatAxis, getaxes(ca)))
function Base.length(ca::ComponentArray) 
    # vca2 = vcat(ca2', ca2') #has not length - is it a valid ComponentVector
    # or rather a Vector<ComponentVector>
    _hasNullOrFlatAxis(ca) && return(length(getdata(ca)))
    prod(length.(getaxes(ca)))
end
function Base.size(ca::ComponentArray) 
    _hasNullOrFlatAxis(ca) && return(size(getdata(ca)))
    map(length, getaxes(ca))
end

# Reduce singleton dimensions
remove_nulls() = ()
remove_nulls(x1, args...) = (x1, remove_nulls(args...)...)
remove_nulls(::NullAxis, args...) = (remove_nulls(args...)...,)

## Attributes
"""
    getdata(x::ComponentArray)

Access `.data` field of a `ComponentArray`, which contains the array that `ComponentArray` wraps.
"""
@inline getdata(x::ComponentArray) = getfield(x, :data)
@inline getdata(x) = x
@inline getdata(x::Adjoint) = getdata(x.parent)'
@inline getdata(x::Transpose) = transpose(getdata(x.parent))

"""
    getaxes(x::ComponentArray)

Access `.axes` field of a `ComponentArray`. This is different than `axes(x::ComponentArray)`, which
returns the axes of the contained array.

# Examples

```jldoctest
julia> using ComponentArrays

julia> ax = Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3)))
Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3)))

julia> A = zeros(6, 6);

julia> ca = ComponentArray(A, (ax, ax))
6×6 ComponentMatrix{Float64} with axes Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3))) × Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3)))
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> getaxes(ca)
(Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3))), Axis(a = 1:3, b = (4:6, (a = 1, b = 2:3))))
```
"""
@inline getaxes(x::ComponentArray) = getfield(x, :axes)
@inline getaxes(x::AdjOrTrans{
    T, <:ComponentVector}) where {T} = (FlatAxis(), getaxes(x.parent)[1])
@inline getaxes(x::AdjOrTrans{T, <:ComponentMatrix}) where {T} = reverse(getaxes(x.parent))

@inline getaxes(::Type{<:ComponentArray{
    T, N, A, Axes}}) where {T, N, A, Axes} = map(x->x(), (Axes.types...,))
@inline getaxes(::Type{<:AdjOrTrans{
    T, CA}}) where {T, CA <: ComponentVector} = (FlatAxis(), getaxes(CA)[1]) |> typeof
@inline getaxes(::Type{<:AdjOrTrans{
    T, CA}}) where {T, CA <: ComponentMatrix} = reverse(getaxes(CA)) |> typeof

## Field access through these functions to reserve dot-getting for keys
@inline getaxes(x::VarAxes) = getaxes(typeof(x))
@inline getaxes(Ax::Type{Axes}) where {Axes <: VarAxes} = map(x->x(), (Ax.types...,))

getaxes(x) = ()

"""
    valkeys(x::ComponentVector)
    valkeys(x::AbstractAxis)

Returns `Val`-wrapped keys of `ComponentVector` for fast iteration over component keys. Also works
directly on an `AbstractAxis`.

# Examples

```jldoctest
julia> using ComponentArrays

julia> ca = ComponentArray(a = 1, b = [1, 2, 3], c = (a = 4,))
ComponentVector{Int64}(a = 1, b = [1, 2, 3], c = (a = 4))

julia> [ca[k] for k in valkeys(ca)]
3-element Vector{Any}:
 1
  [1, 2, 3]
  ComponentVector{Int64}(a = 4)

julia> sum(prod(ca[k]) for k in valkeys(ca))
11
```
"""
@generated function valkeys(ax::AbstractAxis)
    idxmap = indexmap(ax)
    k = Val.(keys(idxmap))
    return :($k)
end
valkeys(ca::ComponentVector) = valkeys(getaxes(ca)[1])
