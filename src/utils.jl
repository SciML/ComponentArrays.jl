@deprecate fastindices(i::Tuple) Val.(i)
@deprecate fastindices(i...) Val.((i...,))

# Make a Val if input isn't already one
toval(x::Val) = x
toval(x) = Val(x)
toval(x::String) = Val(Symbol(x))
toval(x::AbstractArray{Symbol}) = Val((x...,))

# Get value from Val type
getval(::Val{x}) where {x} = x
getval(::Type{Val{x}}) where {x} = x

# Split an array up into partitions where the Ns are partition sizes on each dimension
partition(A) = A
partition(a, N...) = partition(a, N)
# Faster method for vectors and matrices
partition(v, N) = (view(v, i:(i + N - 1)) for i in firstindex(v):N:lastindex(v))
function partition(m, N1, N2)
    ax = axes(m)
    firsts = firstindex.(ax)
    lasts = lastindex.(ax)
    return (
        view(m, i:(i + N1 - 1), j:(j + N2 - 1)) for i in firsts[1]:N1:lasts[1],
            j in firsts[2]:N2:lasts[2]
    )
end
# Slower fallback for higher dimensions
function partition(a::A, N::Tuple) where {A <: AbstractArray}
    ax = axes(a)
    offs = firstindex.(ax)
    return (
        view(a, (:).((I.I .- 1) .* N .+ offs, ((I.I .- 1) .* N .+ N .- 1 .+ offs))...) for I in
            CartesianIndices(
                div.(
                    size(a), N
                )
            )
    )
end
# partition(a::A, N::Tuple) where A<:AbstractVector = reshape(view(a, :), N)

# Faster filtering of tuples by type
filter_by_type(::Type{T}, args...) where {T} = filter_by_type(T, (), args...)
filter_by_type(::Type{T}, part::Tuple) where {T} = part
function filter_by_type(::Type{T}, part::Tuple, ax, args...) where {T}
    return filter_by_type(T, part, args...)
end
function filter_by_type(::Type{T}, part::Tuple, ax::T, args...) where {T}
    return filter_by_type(T, (part..., ax), args...)
end

# Flat length of an arbitrarily nested named tuple
recursive_length(x) = length(x)
recursive_length(a::AbstractArray{T, N}) where {T <: Number, N} = length(a)
recursive_length(a::AbstractArray) = mapreduce(recursive_length, +, a; init = 0)
recursive_length(nt::NamedTuple) = mapreduce(recursive_length, +, values(nt); init = 0)
recursive_length(::Union{Nothing, Missing}) = 1
recursive_length(nt::NamedTuple{(), Tuple{}}) = 0

# Find the highest element type
function recursive_eltype(nt::NamedTuple)
    return isempty(nt) ? Base.Bottom : mapreduce(recursive_eltype, promote_type, nt)
end
function recursive_eltype(x::AbstractArray{<:Any})
    return isempty(x) ? Base.Bottom : mapreduce(recursive_eltype, promote_type, x)
end
function recursive_eltype(x::Dict)
    return isempty(x) ? Base.Bottom : mapreduce(recursive_eltype, promote_type, values(x))
end
recursive_eltype(::AbstractArray{T, N}) where {T <: Number, N} = T
recursive_eltype(x) = typeof(x)

@inline __value(x) = x
