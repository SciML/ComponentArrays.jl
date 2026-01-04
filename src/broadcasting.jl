function Base.BroadcastStyle(::Type{<:ComponentArray{T, N, A, Axes}}) where {T, N, A, Axes}
    return Broadcast.BroadcastStyle(A)
end

# Need special case here for adjoint vectors in order to avoid type instability in axistype
function Broadcast.combine_axes(a::ComponentArray, b::AdjOrTransComponentVector)
    return (axes(a)[1], axes(b)[2])
end
function Broadcast.combine_axes(a::AdjOrTransComponentVector, b::ComponentArray)
    return (axes(b)[2], axes(a)[1])
end

Broadcast.axistype(a::CombinedAxis, b::AbstractUnitRange) = a
Broadcast.axistype(a::AbstractUnitRange, b::CombinedAxis) = b
function Broadcast.axistype(a::CombinedAxis, b::CombinedAxis)
    return CombinedAxis(FlatAxis(), Base.Broadcast.axistype(_array_axis(a), _array_axis(b)))
end
Broadcast.axistype(a::T, b::T) where {T <: CombinedAxis} = a

function Base.promote_shape(
        a::NTuple{M, CombinedAxis}, b::NTuple{
            N, AbstractUnitRange,
        }
    ) where {M, N}
    return Base.promote_shape(_array_axis.(a), b)
end
function Base.promote_shape(
        a::NTuple{N, AbstractUnitRange}, b::NTuple{
            M, CombinedAxis,
        }
    ) where {M, N}
    return Base.promote_shape(a, _array_axis.(b))
end
function Base.promote_shape(
        a::NTuple{M, CombinedAxis}, b::NTuple{
            N, CombinedAxis,
        }
    ) where {M, N}
    return Base.promote_shape(_array_axis.(a), _array_axis.(b))
end
Base.promote_shape(a::T, b::T) where {T <: NTuple{N, CombinedAxis} where {N}} = a

# From https://github.com/JuliaArrays/OffsetArrays.jl/blob/master/src/OffsetArrays.jl
Base.dataids(A::ComponentArray) = Base.dataids(parent(A))
function Broadcast.broadcast_unalias(dest::ComponentArray, src)
    return getdata(dest) === getdata(src) ? src : Broadcast.unalias(dest, src)
end
