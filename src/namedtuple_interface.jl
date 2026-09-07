Base.hash(x::ComponentArray, h::UInt) = hash(keys(x), hash(getdata(x), h))

function Base.:(==)(x::ComponentArray, y::ComponentArray)
    return getdata(x) == getdata(y) && getaxes(x) == getaxes(y)
end
Base.:(==)(x::ComponentArray, y::AbstractArray) = getdata(x) == y && keys(x) == keys(y) # For equality with LabelledArrays
Base.:(==)(x::AbstractArray, y::ComponentArray) = y == x

Base.keys(x::ComponentVector) = keys(indexmap(getaxes(x)[1]))

Base.haskey(x::ComponentVector, s::Symbol) = haskey(indexmap(getaxes(x)[1]), s)

Base.propertynames(x::ComponentVector) = propertynames(indexmap(getaxes(x)[1]))

# Property access for ComponentVectors goes through _get/_setindex
@inline Base.getproperty(x::ComponentVector, s::Symbol) = _getindex(Base.maybeview, x, Val(s))
@inline Base.getproperty(x::ComponentVector, s::Val) = _getindex(Base.maybeview, x, s)

@inline Base.setproperty!(x::ComponentVector, s::Symbol, v) = _setindex!(x, v, Val(s))
@inline Base.setproperty!(x::ComponentVector, s::Val, v) = _setindex!(x, v, s)

Base.merge(a::NamedTuple, b::ComponentArray) = merge(a, NamedTuple(b))
