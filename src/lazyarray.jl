"""
    LazyArray(gen::Base.Generator)

Wrapper around Base.Generator that also indexes like an array. This is needed to make ComponentArrays
that hold arrays of ComponentArrays
"""
struct LazyArray{T, N, G} <: AbstractArray{T, N}
    gen::G
    LazyArray{T}(gen) where {T} = new{T, ndims(gen), typeof(gen)}(gen)
    function LazyArray(gen::Base.Generator{A, F}) where {A, F}
        return new{eltype(A), ndims(gen), typeof(gen)}(gen)
    end
end

const LazyVector{T, G} = LazyArray{T, 1, G}
const LazyMatrix{T, G} = LazyArray{T, 2, G}

Base.getindex(a::LazyArray, i...) = _un_iter(getfield(a, :gen), i)

function Base.setindex!(a::LazyArray, val, i...)
    return a[i...] .= val
end

_un_iter(iter, idxs) = _un_iter(iter.f, iter.iter, idxs)
_un_iter(f, iter::Base.Generator, idxs) = f(_un_iter(iter.f, iter.iter, idxs))
_un_iter(f, iter::Base.Iterators.ProductIterator, idxs) = f(getindex.(iter.iterators, idxs))
_un_iter(f, iter, idxs) = f(iter[idxs...])

Base.getproperty(a::LazyArray, s::Symbol) = LazyArray(getproperty(item, s) for item in a)

Base.propertynames(a::LazyArray) = propertynames(first(a))

Base.keys(a::LazyArray) = Base.OneTo(length(a))

Base.haskey(a::LazyArray, i::Integer) = i in keys(a)

Base.iterate(a::LazyArray) = iterate(getfield(a, :gen))
Base.iterate(a::LazyArray, state...) = iterate(getfield(a, :gen), state...)

Base.collect(a::LazyArray) = collect(getfield(a, :gen))

Base.length(a::LazyArray) = length(getfield(a, :gen))

Base.size(a::LazyArray) = size(getfield(a, :gen))

Base.eltype(::LazyArray{T, N, G}) where {T, N, G} = T

Base.show(io::IO, a::LazyArray) = show(io, collect(a))
function Base.show(io::IO, mime::MIME"text/plain", a::LazyArray)
    arr = collect(a)
    rep = repr(mime, arr)
    return print(io, replace(rep, r"(\d+-element )?((Vector|Array){(.+)?})" => s"\1LazyArray{\4}"; count = 1))
end
