using ComponentArrays
using BenchmarkTools
using ForwardDiff
using Tracker
using InvertedIndices
using LabelledArrays
using LinearAlgebra
using StaticArrays
using OffsetArrays
using Unitful
using Functors
using Test

# Convert abstract unit range to a ViewAxis with ShapeAxis.
r2v(r::AbstractUnitRange) = ViewAxis(r, ShapedAxis(size(r)))

## Test setup
c = (a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45])
nt = (; a = 100, b = [4, 1.3], c)
nt2 = (
    a = 5, b = [(a = (a = 20, b = 1), b = 0), (a = (a = 33, b = 1), b = 0)],
    c = (a = (a = 2, b = [1, 2]), b = [1.0 2.0; 5 6]),
)

ax = Axis(
    a = 1, b = r2v(2:3), c = ViewAxis(
        4:10, (
            a = ViewAxis(1:3, (a = 1, b = r2v(2:3))), b = r2v(4:7),
        )
    )
)
ax_c = (a = ViewAxis(1:3, (a = 1, b = r2v(2:3))), b = r2v(4:7))

a = Float64[100, 4, 1.3, 1, 1, 4.4, 0.4, 2, 1, 45]
sq_mat = collect(reshape(1:9, 3, 3))

ca = ComponentArray(nt)
ca_Float32 = ComponentArray{Float32}(nt)
ca_MVector = ComponentArray{MVector{10, Float64}}(nt) # TODO: Deprecate these
ca_SVector = ComponentArray{SVector{10, Float64}}(nt)
ca_composed = ComponentArray(a = 1, b = ca)

ca2 = ComponentArray(nt2)

cmat = ComponentArray(a .* a', ax, ax)
cmat2 = ca2 .* ca2'

caa = ComponentArray(a = ca, b = sq_mat)

_a, _b, _c = Val.((:a, :b, :c))

ca3 = ComponentArray(a = 1, b = [2, 3, 4, 5], c = reshape(6:11, 3, 2))
cmat3 = ca3 .* ca3'
cmat3check = (1:11) .* (1:11)'

## Tests
