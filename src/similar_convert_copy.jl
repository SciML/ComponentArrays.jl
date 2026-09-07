const CombinedAnyDims = Tuple{<:CombinedAxis, Vararg{CombinedOrRegularAxis}}
const AnyCombinedAnyDims = Tuple{
    <:CombinedOrRegularAxis, <:CombinedAxis, Vararg{CombinedOrRegularAxis},
}
const CombinedCombinedAnyDims = Tuple{
    <:CombinedAxis, <:CombinedAxis, Vararg{CombinedOrRegularAxis},
}

# Similar
Base.similar(x::ComponentArray) = ComponentArray(similar(getdata(x)), getaxes(x)...)
function Base.similar(x::ComponentArray, ::Type{T}) where {T}
    return ComponentArray(similar(getdata(x), T), getaxes(x)...)
end
Base.similar(x::ComponentArray, dims::Vararg{Int}) = similar(getdata(x), dims...)
function Base.similar(x::ComponentArray, ::Type{T}, dims::Vararg{Int}) where {T}
    return similar(getdata(x), T, dims...)
end
Base.similar(x::AbstractArray, dims::CombinedAnyDims) = _similar(x, dims)
Base.similar(x::AbstractArray, dims::AnyCombinedAnyDims) = _similar(x, dims)
Base.similar(x::AbstractArray, dims::CombinedCombinedAnyDims) = _similar(x, dims)
function Base.similar(x::AbstractArray, ::Type{T}, dims::CombinedAnyDims) where {T}
    return _similar(x, T, dims)
end
function Base.similar(x::AbstractArray, ::Type{T}, dims::AnyCombinedAnyDims) where {T}
    return _similar(x, T, dims)
end
function Base.similar(x::AbstractArray, ::Type{T}, dims::CombinedCombinedAnyDims) where {T}
    return _similar(x, T, dims)
end
Base.similar(x::Type{<:AbstractArray}, dims::CombinedAnyDims) = _similar(x, dims)
Base.similar(x::Type{<:AbstractArray}, dims::AnyCombinedAnyDims) = _similar(x, dims)
Base.similar(x::Type{<:AbstractArray}, dims::CombinedCombinedAnyDims) = _similar(x, dims)

function _similar(x::AbstractArray, dims)
    return ComponentArray(similar(getdata(x), length.(_array_axis.(dims))), _component_axis.(dims)...)
end
function _similar(x::Type, dims)
    return ComponentArray(similar(x, length.(_array_axis.(dims))), _component_axis.(dims)...)
end
function _similar(x, T, dims)
    return ComponentArray(similar(getdata(x), T, length.(_array_axis.(dims))), _component_axis.(dims)...)
end

Base.zero(x::ComponentArray) = ComponentArray(zero(getdata(x)), getaxes(x)...)

## FIXME: waiting on similar(::Type{<:ComponentArray})
# Base.zeros(CA::Type{<:ComponentArray}) = (similar(CA) .= 0)

# Base.ones(CA::Type{<:ComponentArray}) = (similar(CA) .= 1)

# Copy
Base.copy(x::ComponentArray) = ComponentArray(copy(getdata(x)), getaxes(x))

Base.copyto!(dest::AbstractArray, src::ComponentArray) = copyto!(dest, getdata(src))
function Base.copyto!(dest::ComponentArray, src::AbstractArray)
    copyto!(getdata(dest), src)
    return dest
end
function Base.copyto!(dest::ComponentArray, src::ComponentArray)
    copyto!(getdata(dest), getdata(src))
    return dest
end

Base.deepcopy(x::ComponentArray) = ComponentArray(deepcopy(getdata(x)), getaxes(x))

function Base.convert(::Type{ComponentArray{T, N, AA, Ax}}, A::AbstractArray) where {
        T, N, AA, Ax,
    }
    return ComponentArray{Ax}(A)
end
function Base.convert(
        ::Type{ComponentArray{T, N, A, Ax1}},
        x::ComponentArray{T, N, A, Ax2}
    ) where {T, N, A, Ax1, Ax2}
    return x
end
function Base.convert(
        ::Type{ComponentArray{T1, N, A1, Ax1}},
        x::ComponentArray{T2, N, A2, Ax2}
    ) where {T1, T2, N, A1, A2, Ax1, Ax2}
    return T1.(x)
end
function Base.convert(
        ::Type{ComponentArray{T, N, A1, Ax}},
        x::ComponentArray{T, N, A2, Ax}
    ) where {T, N, A1, A2, Ax}
    return x
end
function Base.convert(
        ::Type{ComponentArray{T, N, A, Ax}}, x::ComponentArray{
            T, N, A, Ax,
        }
    ) where {T, N, A, Ax}
    return x
end
Base.convert(T::Type{<:Array}, x::ComponentArray) = convert(T, getdata(x))

function Base.convert(
        ::Type{Cholesky{T1, Matrix{T1}}}, x::Cholesky{
            T2, <:ComponentArray,
        }
    ) where {T1, T2}
    return Cholesky(Matrix{T1}(x.factors), x.uplo, x.info)
end

# Conversion to from ComponentArray to NamedTuple (note, does not preserve numeric types of
# original NamedTuple)
function _namedtuple(x::ComponentVector)
    return NamedTuple{keys(x)}(
        map(valkeys(x)) do key
            _namedtuple(getproperty(x, key))
        end
    )
end
_namedtuple(v::AbstractVector) = _namedtuple.(v)
_namedtuple(x) = x

Base.convert(::Type{NamedTuple}, x::ComponentVector) = _namedtuple(x)
Base.NamedTuple(x::ComponentVector) = _namedtuple(x)

## AbstractAxis conversion and promotion
Base.convert(::Type{Ax}, ax::AbstractAxis) where {Ax <: AbstractAxis} = ax
