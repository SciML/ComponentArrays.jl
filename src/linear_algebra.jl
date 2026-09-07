## https://github.com/SciML/DifferentialEquations.jl/issues/849
function ArrayInterface.lu_instance(x::ComponentMatrix)
    T = eltype(x)
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LU{luT}(similar(x), ipiv, info)
end

# The generic `lu`/`qr` copy through `similar(A, T, dims)`, which drops the axes
# and returns a plain `Matrix`, whereas the in-place `lu!`/`qr!` (ComponentMatrix
# is strided, so they hit LAPACK directly) keep the ComponentMatrix wrapper on the
# factors. Make the out-of-place versions and the `ArrayInterface` instance
# functions preserve the wrapper too, so callers that type a cache from the
# instance functions and later store either variant (e.g. LinearSolve) don't hit
# a cache type mismatch.
function LinearAlgebra.lu(
        A::ComponentMatrix, pivot::Union{NoPivot, RowNonZero, RowMaximum} = RowMaximum();
        kwargs...,
    )
    F = LinearAlgebra.lu(getdata(A), pivot; kwargs...)
    return LU(ComponentArray(F.factors, getaxes(A)), F.ipiv, F.info)
end

function LinearAlgebra.qr(
        A::ComponentMatrix, pivot::Union{NoPivot, ColumnNorm} = NoPivot();
        kwargs...,
    )
    F = LinearAlgebra.qr(getdata(A), pivot; kwargs...)
    return _rewrap_qr(F, getaxes(A))
end

function ArrayInterface.qr_instance(A::ComponentMatrix, pivot = NoPivot())
    F = ArrayInterface.qr_instance(getdata(A), pivot)
    return _rewrap_qr(F, getaxes(A))
end

# QRCompactWY is not public LinearAlgebra API; allow-listed in test/qa/qa.jl.
_rewrap_qr(F::LinearAlgebra.QRCompactWY, ax) = LinearAlgebra.QRCompactWY(ComponentArray(F.factors, ax), F.T)
_rewrap_qr(F::LinearAlgebra.QRPivoted, ax) = LinearAlgebra.QRPivoted(ComponentArray(F.factors, ax), F.τ, F.jpvt)
_rewrap_qr(F::LinearAlgebra.QR, ax) = LinearAlgebra.QR(ComponentArray(F.factors, ax), F.τ)
_rewrap_qr(F, ax) = F

# Helpers for dealing with adjoints and such
_first_axis(x::AbstractComponentVecOrMat) = getaxes(x)[1]

_second_axis(x::AbstractMatrix) = FlatAxis()
_second_axis(x::ComponentMatrix) = getaxes(x)[2]
_second_axis(x::AdjOrTransComponentVecOrMat) = getaxes(x)[2]

_out_axes(::typeof(*), a, b::AbstractVector) = (_first_axis(a),)
_out_axes(::typeof(*), a, b::AbstractMatrix) = (_first_axis(a), _second_axis(b))
_out_axes(::typeof(\), a, b::AbstractVector) = (_second_axis(a),)
_out_axes(::typeof(\), a, b::AbstractMatrix) = (_second_axis(a), _second_axis(b))
_out_axes(::typeof(/), a::AbstractMatrix, b) = (_first_axis(a), _first_axis(b))

# Arithmetic
for op in [:*, :\, :/]
    @eval begin
        function Base.$op(A::AbstractComponentVecOrMat, B::AbstractComponentVecOrMat)
            C = $op(getdata(A), getdata(B))
            ax = _out_axes($op, A, B)
            return ComponentArray(C, ax)
        end
    end
    for (adj, Adj) in zip([:adjoint, :transpose], [:Adjoint, :Transpose])
        @eval begin
            function Base.$op(aᵀ::$Adj{T, <:ComponentVector}, B::AbstractComponentMatrix) where {T}
                cᵀ = $op(getdata(aᵀ), getdata(B))
                ax2 = _out_axes($op, aᵀ, B)[2]
                return $adj(ComponentArray(cᵀ', ax2))
            end
            function Base.$op(A::$Adj{T, <:CV}, B::CV) where {
                    T <: Real, CV <: ComponentVector{T},
                }
                return $op(getdata(A), getdata(B))
            end
        end
    end
end

# Common Accumulation Operations
## Needed for CUDA to work properly
function LinearAlgebra.axpy!(α::Number, x::ComponentArray, y::ComponentArray)
    getaxes(x) != getaxes(y) && throw(ArgumentError("Axes of `x` and `y` must match"))
    axpy!(α, getdata(x), getdata(y))
    return y
end

function LinearAlgebra.axpby!(α::Number, x::ComponentArray, β::Number, y::ComponentArray)
    getaxes(x) != getaxes(y) && throw(ArgumentError("Axes of `x` and `y` must match"))
    axpby!(α, getdata(x), β, getdata(y))
    return y
end

function LinearAlgebra.ldiv!(B::AbstractVecOrMat, D::Diagonal{Float64, <:ComponentArray}, A::AbstractVecOrMat)
    return ldiv!(B, Diagonal(Vector(D.diag)), A)
end
