Base.parent(x::ComponentArray) = getfield(x, :data)

Base.size(x::ComponentArray) = size(getdata(x))
function StaticArrayInterface.static_size(A::ComponentArray)
    return StaticArrayInterface.static_size(parent(A))
end

Base.elsize(x::Type{<:ComponentArray{T, N, A, Axes}}) where {T, N, A, Axes} = Base.elsize(A)

# Base.axes(x::ComponentArray) = axes(getdata(x))
Base.axes(x::ComponentArray) = CombinedAxis.(getaxes(x), axes(getdata(x)))

function Base.reinterpret(::Type{T}, x::ComponentArray, args...) where {T}
    return ComponentArray(reinterpret(T, getdata(x), args...), getaxes(x))
end

function ArrayInterface.indices_do_not_alias(
        ::Type{
            ComponentArray{
                T, N, A, Axes,
            },
        }
    ) where {T, N, A, Axes}
    return ArrayInterface.indices_do_not_alias(A)
end
function ArrayInterface.instances_do_not_alias(
        ::Type{
            ComponentArray{
                T, N, A, Axes,
            },
        }
    ) where {T, N, A, Axes}
    return ArrayInterface.instances_do_not_alias(A)
end

# Cats
# TODO: Make this a little less copy-pastey
function Base.hcat(x::AbstractComponentVecOrMat, y::AbstractComponentVecOrMat)
    ax_x, ax_y = second_axis.((x, y))
    if reduce((accum, key) -> accum || (key in keys(ax_x)), keys(ax_y); init = false) ||
            getaxes(x)[1] != getaxes(y)[1]
        return hcat(getdata(x), getdata(y))
    else
        data_x, data_y = getdata.((x, y))
        ax_y = reindex(ax_y, size(x, 2))
        idxmap_x, idxmap_y = indexmap.((ax_x, ax_y))
        axs = getaxes(x)
        return ComponentArray(
            hcat(data_x, data_y), axs[1], Axis(
                (;
                    idxmap_x..., idxmap_y...,
                )
            ), axs[3:end]...
        )
    end
end

second_axis(ca::AbstractComponentVecOrMat) = getaxes(ca)[2]
second_axis(::ComponentVector) = FlatAxis()

# Are all these methods necessary?
# TODO: See what we can reduce down to without getting ambiguity errors
Base.vcat(x::ComponentVector{<:Number}, y::AbstractVector{<:Number}) = vcat(getdata(x), y)
Base.vcat(x::AbstractVector{<:Number}, y::ComponentVector{<:Number}) = vcat(x, getdata(y))
function Base.vcat(x::ComponentVector{<:Number}, y::ComponentVector{<:Number})
    if reduce((accum, key) -> accum || (key in keys(x)), keys(y); init = false)
        return vcat(getdata(x), getdata(y))
    else
        data_x, data_y = getdata.((x, y))
        ax_x, ax_y = getindex.(getaxes.((x, y)), 1)
        ax_y = reindex(ax_y, length(x))
        idxmap_x, idxmap_y = indexmap.((ax_x, ax_y))
        return ComponentArray(vcat(data_x, data_y), Axis((; idxmap_x..., idxmap_y...)))
    end
end
function Base.vcat(x::AbstractComponentVecOrMat{<:Number}, y::AbstractComponentVecOrMat{<:Number})
    ax_x, ax_y = getindex.(getaxes.((x, y)), 1)
    if reduce((accum, key) -> accum || (key in keys(ax_x)), keys(ax_y); init = false) ||
            getaxes(x)[2:end] != getaxes(y)[2:end]
        return vcat(getdata(x), getdata(y))
    else
        data_x, data_y = getdata.((x, y))
        ax_y = reindex(ax_y, size(x, 1))
        idxmap_x, idxmap_y = indexmap.((ax_x, ax_y))
        return ComponentArray(vcat(data_x, data_y), Axis((; idxmap_x..., idxmap_y...)), getaxes(x)[2:end]...)
    end
end
function Base.vcat(x::CV...) where {CV <: AdjOrTransComponentArray{<:Number}}
    return ComponentArray(reduce(vcat, map(y -> getdata(y.parent)', x)), getaxes(x[1]))
end
Base.vcat(x::ComponentVector{<:Number}, args...) = vcat(getdata(x), getdata.(args)...)
function Base.vcat(
        x::ComponentVector{<:Number},
        args::Vararg{Union{Number, UniformScaling, AbstractVecOrMat{<:Number}}}
    )
    return vcat(getdata(x), getdata.(args)...)
end
function Base.vcat(
        x::ComponentVector{<:Number}, args::Vararg{
            AbstractVector{T}, N,
        }
    ) where {T <: Number, N}
    return vcat(getdata(x), getdata.(args)...)
end

function Base.hvcat(row_lengths::NTuple{N, Int}, xs::Vararg{AbstractComponentVecOrMat}) where {N}
    i = 1
    idxs = UnitRange{Int}[]
    for row_length in row_lengths
        i_last = i + row_length - 1
        push!(idxs, i:i_last)
        i = i_last + 1
    end
    rows = [reduce(hcat, xs[idx]) for idx in idxs]
    return vcat(rows...)
end

function Base.permutedims(x::ComponentArray, dims)
    axs = getaxes(x)
    return ComponentArray(permutedims(getdata(x), dims), map(i -> axs[i], dims)...)
end

## Indexing
function Base.IndexStyle(::Type{<:ComponentArray{T, N, <:A, <:Axes}}) where {T, N, A, Axes}
    return IndexStyle(A)
end

# Since we aren't really using the standard approach to indexing, this will forward things to
# the correct methods
Base.to_indices(x::ComponentArray, i::Tuple{Any}) = i
function Base.to_indices(
        x::ComponentArray, i::NTuple{
            N, Union{Integer, CartesianIndex},
        }
    ) where {N}
    return i
end
Base.to_indices(x::ComponentArray, i::NTuple{N, Int64}) where {N} = i
Base.to_index(x::ComponentArray, i) = i

# Get ComponentArray index
Base.@propagate_inbounds Base.getindex(x::ComponentArray, idx::CartesianIndex) = getdata(x)[idx]
Base.@propagate_inbounds Base.getindex(x::ComponentArray, idx::Vararg{FlatIdx}) = getdata(x)[idx...]
Base.@propagate_inbounds function Base.getindex(x::ComponentArray, idx::Vararg{FlatOrColonIdx})
    axs = map((ax, i) -> getindex(ax, i).ax, getaxes(x), idx)
    axs = remove_nulls(axs...)
    return ComponentArray(getdata(x)[idx...], axs...)
end
Base.@propagate_inbounds Base.getindex(x::ComponentArray, ::Colon) = getdata(x)[:]
Base.@propagate_inbounds Base.getindex(x::ComponentArray, ::Colon, ::Vararg{Colon}) = x
@inline Base.getindex(x::ComponentArray, idx...) = getindex(x, toval.(idx)...)
@inline Base.getindex(x::ComponentArray, idx::Vararg{Val}) = _getindex(getindex, x, idx...)

# Set ComponentArray index
Base.@propagate_inbounds Base.setindex!(
    x::ComponentArray, v, idx::FlatOrColonIdx...
) = setindex!(getdata(x), v, idx...)
Base.@propagate_inbounds Base.setindex!(x::ComponentArray, v, ::Colon) = setindex!(getdata(x), v, :)
@inline Base.setindex!(x::ComponentArray, v, idx...) = setindex!(x, v, toval.(idx)...)
@inline Base.setindex!(x::ComponentArray, v, idx::Vararg{Val}) = _setindex!(x, v, idx...)

# Explicitly view
Base.@propagate_inbounds Base.view(
    x::ComponentArray, idx::Vararg{ComponentArrays.FlatIdx}
) = view(getdata(x), idx...)
Base.@propagate_inbounds Base.view(x::ComponentArray, idx...) = _getindex(view, x, toval.(idx)...)

Base.@propagate_inbounds Base.maybeview(
    x::ComponentArray, idx::Vararg{ComponentArrays.FlatIdx}
) = Base.maybeview(getdata(x), idx...)
Base.@propagate_inbounds Base.maybeview(
    x::ComponentArray, idx...
) = _getindex(Base.maybeview, x, toval.(idx)...)

# Generated get and set index methods to do all of the heavy lifting in the type domain
@generated function _getindex(index_fun, x::ComponentArray, idx...)
    ci = getindex.(getaxes(x), getval.(idx))
    inds = map(i -> i.idx, ci)
    axs = map(i -> i.ax, ci)
    axs = remove_nulls(axs...)
    if isa(axs, Tuple{})
        return :(index_fun(getdata(x), $inds...))
    else
        return :(ComponentArray(index_fun(getdata(x), $inds...), $axs...))
    end
end

@generated function _setindex!(x::ComponentArray, v, idx...)
    ci = getindex.(getaxes(x), getval.(idx))
    inds = map(i -> i.idx, ci)
    return :(setindex!(getdata(x), v, $inds...))
end

## Linear Algebra
function Base.pointer(x::ComponentArray{T, N, A, Axes}) where {T, N, A <: DenseArray, Axes}
    return pointer(getdata(x))
end

function Base.unsafe_convert(
        ::Type{Ptr{T}}, x::ComponentArray{
            T, N, A, Axes,
        }
    ) where {T, N, A, Axes}
    return Base.unsafe_convert(Ptr{T}, getdata(x))
end

Base.strides(x::ComponentArray) = strides(getdata(x))
for f in [:device, :stride_rank, :contiguous_axis, :contiguous_batch_size, :dense_dims]
    @eval StaticArrayInterface.$f(
        ::Type{
            ComponentArray{
                T, N, A, Axes,
            },
        }
    ) where {T, N, A, Axes} = StaticArrayInterface.$f(A)
end

Base.stride(x::ComponentArray, k) = stride(getdata(x), k)
Base.stride(x::ComponentArray, k::Int64) = stride(getdata(x), k)

ArrayInterface.parent_type(::Type{ComponentArray{T, N, A, Axes}}) where {T, N, A, Axes} = A
