module ComponentArraysGPUArraysExt

using ComponentArrays, LinearAlgebra, GPUArrays
using ComponentArrays: recursive_eltype

const GPUComponentArray = ComponentArray{
    T, N, <:GPUArrays.AbstractGPUArray, Ax,
} where {T, N, Ax}
const GPUComponentVector{T, Ax} = ComponentArray{T, 1, <:GPUArrays.AbstractGPUVector, Ax}
const GPUComponentMatrix{T, Ax} = ComponentArray{T, 2, <:GPUArrays.AbstractGPUMatrix, Ax}
const GPUComponentVecorMat{
    T, Ax,
} = Union{GPUComponentVector{T, Ax}, GPUComponentMatrix{T, Ax}}

@static if pkgversion(GPUArrays) < v"11"
    GPUArrays.backend(x::ComponentArray) = GPUArrays.backend(getdata(x))

    function Base.fill!(A::GPUComponentArray{T}, x) where {T}
        length(A) == 0 && return A
        GPUArrays.gpu_call(A, convert(T, x)) do ctx, a, val
            idx = GPUArrays.@linearidx(a)
            @inbounds a[idx] = val
            return
        end
        return A
    end
else
    function Base.fill!(A::GPUComponentArray{T}, x) where {T}
        length(A) == 0 && return A
        ComponentArrays.fill_componentarray_ka!(A, x)
        return A
    end
end

LinearAlgebra.dot(x::GPUComponentArray, y::GPUComponentArray) = dot(getdata(x), getdata(y))
LinearAlgebra.dot(x::GPUComponentArray, y::AbstractGPUArray) = dot(getdata(x), y)
LinearAlgebra.dot(x::AbstractGPUArray, y::GPUComponentArray) = dot(x, getdata(y))

LinearAlgebra.norm(ca::GPUComponentArray, p::Real) = norm(getdata(ca), p)
LinearAlgebra.rmul!(ca::GPUComponentArray, b::Number) = GPUArrays.generic_rmul!(ca, b)

function Base.map(f, x::GPUComponentArray, args...)
    data = map(f, getdata(x), getdata.(args)...)
    return ComponentArray(data, getaxes(x))
end
function Base.map(
        f, x::GPUComponentArray, args::Vararg{
            Union{
                Base.AbstractBroadcasted, AbstractArray,
            },
        }
    )
    data = map(f, getdata(x), map(getdata, args)...)
    return ComponentArray(data, getaxes(x))
end

# We need all of these to avoid method ambiguities
function Base.mapreduce(f, op, x::GPUComponentArray; kwargs...)
    return mapreduce(f, op, getdata(x); kwargs...)
end
function Base.mapreduce(f, op, x::GPUComponentArray, args...; kwargs...)
    return mapreduce(f, op, getdata(x), map(getdata, args)...; kwargs...)
end
function Base.mapreduce(
        f, op, x::GPUComponentArray,
        args::Vararg{Union{Base.AbstractBroadcasted, AbstractArray}}; kwargs...
    )
    return mapreduce(f, op, getdata(x), map(getdata, args)...; kwargs...)
end

# These are all stolen from GPUArrays.j;
Base.any(A::GPUComponentArray{Bool}) = mapreduce(identity, |, getdata(A))
Base.all(A::GPUComponentArray{Bool}) = mapreduce(identity, &, getdata(A))

Base.any(f::Function, A::GPUComponentArray) = mapreduce(f, |, getdata(A))
Base.all(f::Function, A::GPUComponentArray) = mapreduce(f, &, getdata(A))

function Base.count(pred::Function, A::GPUComponentArray; dims = :, init = 0)
    return mapreduce(pred, Base.add_sum, getdata(A); init = init, dims = dims)
end

# avoid calling into `initarray!`
for (fname, op) in [
        (:sum, :(Base.add_sum)), (:prod, :(Base.mul_prod)),
        (:maximum, :(Base.max)), (:minimum, :(Base.min)),
        (:all, :&), (:any, :|),
    ]
    fname! = Symbol(fname, '!')
    @eval begin
        Base.$(fname!)(
            f::Function,
            r::GPUComponentArray,
            A::GPUComponentArray{T}
        ) where {T} = GPUArrays.mapreducedim!(
            f, $(op), getdata(r), getdata(A); init = neutral_element($(op), T)
        )
    end
end

function ComponentArrays.ComponentArray(
        nt::NamedTuple{
            names,
            <:Tuple{Vararg{Union{GPUArrays.AbstractGPUArray, GPUComponentArray}}},
        }
    ) where {names}
    T = recursive_eltype(nt)
    gpuarray = getdata(first(nt))
    G = Base.typename(typeof(gpuarray)).wrapper  # SciMLBase.parameterless_type(gpuarray)
    return GPUArrays.adapt(G, ComponentArray(NamedTuple{names}(map(GPUArrays.adapt(Array{T}), nt))))
end

function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::GPUComponentVecorMat, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end

function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::GPUComponentVecorMat, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::GPUComponentVecorMat, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::GPUComponentVecorMat, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, B::GPUComponentVecorMat,
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        },
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        },
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Number, b::Number
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end

function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::GPUComponentVecorMat, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat}, a::Real,
        b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::GPUComponentVecorMat,
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::GPUComponentVecorMat, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::GPUComponentVecorMat, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::GPUComponentVecorMat, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, B::GPUComponentVecorMat,
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat}, a::Real,
        b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        },
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUVecOrMat}, a::Real,
        b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        B::LinearAlgebra.Adjoint{<:Any, <:GPUComponentVecorMat},
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        B::LinearAlgebra.Transpose{<:Any, <:GPUArrays.AbstractGPUVecOrMat},
        a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(
        C::GPUComponentVecorMat,
        A::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        },
        B::LinearAlgebra.Transpose{
            <:Any, <:GPUComponentVecorMat,
        }, a::Real, b::Real
    )
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end

end
