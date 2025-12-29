using ComponentArrays
using BenchmarkTools
using ForwardDiff
using Tracker
using InvertedIndices
using LabelledArrays
using LinearAlgebra
using StaticArrays
using OffsetArrays
using Test
using Unitful
using Functors

include("formalities.jl")

# Convert abstract unit range to a ViewAxis with ShapeAxis.
r2v(r::AbstractUnitRange) = ViewAxis(r, ShapedAxis(size(r)))

## Test setup
c = (a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45])
nt = (a = 100, b = [4, 1.3], c = c)
nt2 = (a = 5, b = [(a = (a = 20, b = 1), b = 0), (a = (a = 33, b = 1), b = 0)],
    c = (a = (a = 2, b = [1, 2]), b = [1.0 2.0; 5 6]))

ax = Axis(a = 1, b = r2v(2:3), c = ViewAxis(4:10, (
    a = ViewAxis(1:3, (a = 1, b = r2v(2:3))), b = r2v(4:7))))
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
@testset "Allocations and Inference" begin
    @test @ballocated($ca.c.a.a) == 0
    @test @ballocated(@view $ca[:c]) == 0
    @test @ballocated(@view $cmat[:c, :c]) == 0

    f = (out, x) -> (out .= x .+ x)
    out = deepcopy(ca)
    @test @ballocated($f($out, $ca)) == 0
end

@testset "Utilities" begin
    @test_deprecated ComponentArrays.getval.(fastindices(:a, :b, :c)) == (:a, :b, :c)
    @test_deprecated fastindices(:a, Val(:b)) == (Val(:a), Val(:b))

    @test collect(ComponentArrays.partition(collect(1:12), 3)) ==
          [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    @test size(collect(ComponentArrays.partition(zeros(2, 2, 2), 1, 2, 2))[2, 1, 1]) ==
          (1, 2, 2)
end

@testset "Construction" begin
    @test ca == ComponentArray(a = 100, b = [4, 1.3], c = (
        a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45]))
    @test ca_Float32 == ComponentArray(Float32.(a), ax)
    @test eltype(ComponentArray{ForwardDiff.Dual}(nt)) == ForwardDiff.Dual
    @test ca_composed.b isa ComponentArray
    @test ca_composed.b == ca
    @test getdata(ca_MVector) isa MArray
    @test typeof(ComponentArray(undef, (ax,))) == typeof(ca)
    @test typeof(ComponentArray(undef, (ax, ax))) == typeof(cmat)
    @test typeof(ComponentArray{Float32}(undef, (ax,))) == typeof(ca_Float32)
    @test typeof(ComponentArray{MVector{10, Float64}}(undef, (ax,))) == typeof(ca_MVector)

    # Entry from Dict
    dict1 = Dict(:a => rand(5), :b => rand(5, 5))
    dict2 = Dict(:a => 3, :b => dict1)
    @test ComponentArray(dict1) isa ComponentArray
    @test ComponentArray(dict2).b isa ComponentArray

    @test ca == ComponentVector(a = 100, b = [4, 1.3], c = (
        a = (a = 1, b = [1.0, 4.4]), b = [0.4, 2, 1, 45]))
    @test cmat == ComponentMatrix(a .* a', ax, ax)
    @test_throws DimensionMismatch ComponentVector(sq_mat, ax)
    @test_throws DimensionMismatch ComponentMatrix(rand(11, 11, 11), ax, ax)
    @test_throws ErrorException ComponentArray(v = [(a = 1, b = 2), (a = 3, c = 4)])

    # Axis construction from symbols
    @test Axis([:a, :b, :c]) == Axis(a = 1, b = 2, c = 3)
    @test Axis((:a, :b, :c)) == Axis(a = 1, b = 2, c = 3)
    @test Axis(:a, :b, :c) == Axis(a = 1, b = 2, c = 3)
    @test_throws ErrorException Axis(:a, :a)

    # Issue #24
    @test ComponentVector(a = 1, b = 2.0f0) == ComponentVector{Float32}(a = 1.0, b = 2.0)
    @test ComponentVector(a = 1, b = 2 + im) ==
          ComponentVector{Complex{Int64}}(a = 1 + 0im, b = 2 + 1im)

    # Issue #23
    sz = size(ca)
    temp = ComponentArray(ca; d = 100)
    temp2 = ComponentVector(temp; d = 4)
    temp3 = ComponentArray(temp2; e = (a = 20, b = [2 4; 1 4]))
    @test sz == size(ca)
    @test temp.d == 100
    @test temp2.d == 4
    @test !haskey(ca, :d)
    @test all(temp3.e.b .== [2 4; 1 4])

    # Issue #18
    temp_miss = ComponentArray(a = missing, b = [2, 1, 4, 5], c = [1, 2, 3])
    @test eltype(temp_miss) == Union{Int64, Missing}
    @test temp_miss.a === missing
    temp_noth = ComponentArray(a = nothing, b = [2, 1, 4, 5], c = [1, 2, 3])
    @test eltype(temp_noth) == Union{Int64, Nothing}
    @test temp_noth.a === nothing

    # Issue #61
    @test ComponentArray(x = 1) isa ComponentArray{Int}

    # Issue #81
    @test ComponentArray() isa ComponentArray
    @test ComponentVector() isa ComponentVector
    @test ComponentMatrix() isa ComponentMatrix
    @test ComponentArray{Float32}() isa ComponentArray{Float32}
    @test ComponentVector{Float32}() isa ComponentVector{Float32}
    @test ComponentMatrix{Float32}() isa ComponentMatrix{Float32}

    # Issue #116
    # Part 2: Arrays of arrays
    @test_throws Exception ComponentVector(a = [[3], [4, 5]], b = 1)

    x = ComponentVector(a = [[3, 3], [4, 5]], b = 1)
    @test x.a[1] == [3, 3]
    @test x.b == 1

    # empty components
    for T in [Int64, Int32, Float64, Float32, ComplexF64, ComplexF32]
        @test ComponentArray(a = T[]) == ComponentVector{T}(a = T[])
        @test ComponentArray(a = T[], b = T[]) == ComponentVector{T}(a = T[], b = T[])
        @test ComponentArray(a = T[], b = (;)) == ComponentVector{T}(a = T[], b = T[])
        @test ComponentArray(a = Any[one(Int32)], b = T[]) ==
              ComponentVector{T}(a = [one(T)], b = T[])
    end
    @test ComponentArray(NamedTuple()) == ComponentVector{Any}()
    @test ComponentArray(a = []).a == []

    # Make sure type promotion works correctly with StaticArrays of NamedTuples
    @test ComponentVector(a = SA[(a = 2, b = true)], b = false) isa ComponentVector{Int}
end

@testset "Attributes" begin
    @test length(ca) == length(a)
    @test size(ca) == size(a)
    @test size(cmat) == (length(a), length(a))

    @test propertynames(ca) == (:a, :b, :c)
    @test propertynames(ca.c) == (:a, :b)

    @test parent(ca) == a

    @test keys(ca) == (:a, :b, :c)
    @test valkeys(ca) == Val.((:a, :b, :c))

    @test ca != getdata(ca)
    @test getdata(ca) != ca
    @test hash(ca) != hash(getdata(ca))
    @test hash(ca, zero(UInt)) != hash(getdata(ca), zero(UInt))

    ab = ComponentArray(a = 1, b = 2)
    xy = ComponentArray(x = 1, y = 2)
    @test ab != xy
    @test hash(ab) != hash(xy)
    @test hash(ab, zero(UInt)) != hash(xy, zero(UInt))

    @test ab == LVector(a = 1, b = 2)

    # Issue #117
    kw_fun(; a, b) = a // b
    x = ComponentArray(b = 1, a = 2)
    @test merge(NamedTuple(), x) == NamedTuple(x)
    @test kw_fun(; x...) == 2

    @test length(ViewAxis(2:7, ShapedAxis((2, 3)))) == 6
end

@testset "Get" begin
    @test getdata(ca) == a
    @test getdata(cmat) == a .* a'

    @test getaxes(ca) == (ax,)
    @test getaxes(cmat) == (ax, ax)

    @test ca[1] == a[1]
    @test ca[1:5] == a[1:5]
    @test cmat[:, :] == cmat
    @test getaxes(cmat[:a, :]) == getaxes(ca)

    @test ca.a == 100.0
    @test ca.b == Float64[4, 1.3]
    @test ca.c.a.a == 1.0
    @test ca.c.a.b[1] == 1.0
    @test ca.c == ComponentArray(c)
    @test ca2.b[1].a.a == 20.0

    @test ca[:a] == ca["a"] == ca.a == ca[[:a]][1]
    @test ca[[:a]] isa ComponentVector  # Issue 175
    @test ca[Symbol[]] == Float64[]  # Issue 174
    @test length(ca[()]) == 0  # Issue #174
    @test ca[:b] == ca["b"] == ca.b
    @test ca[:c] == ca["c"] == ca.c

    @test ca[(:a, :c)].c == ca[(:c, :a)].c == ca.c
    @test ca[(:a, :c)].a isa Number
    @test ca[[:a, :c]] == ca[(:a, :c)]
    @test_throws AssertionError ca[(:a, :a)]

    @test cmat[:a, :a] == cmat["a", "a"] == 10000.0
    @test cmat[:a, :b] == cmat["a", "b"] == [400, 130]
    @test all(cmat[:c, :c] .== ComponentArray(a[4:10] .* a[4:10]', Axis(ax_c), Axis(ax_c)))
    @test cmat[:c, :][:a, :][:a, :] == ca
    @test cmat[:a, :c] == cmat[:c, :a]
    @test all(cmat2[:b, :b][1, 1] .== ca2.b[1] .* ca2.b[1]')

    @test ca[_a] == ca[:a]
    @test cmat[_c, _b] == cmat[:c, :b]
    @test cmat[_c, :a] == cmat[:c, :a]

    @test ca2.b[2].a.a == 33

    @test collect(caa.b) == sq_mat
    @test size(caa.b) == size(sq_mat)
    @test caa.b[1:2, 3] == sq_mat[1:2, 3]

    @test Base.maybeview(ca, :a) == ca.a
    @test cmat[:c, :a] == getindex(cmat, :c, :a)
    @test @view(cmat[:c, :a]) == view(cmat, :c, :a)

    @test ca[CartesianIndex(1)] == ca[1]
    @test cmat[CartesianIndex(1, 2)] == cmat[1, 2]
    @test cmat[CartesianIndices(cmat)] == getdata(cmat)

    @test getproperty(ca, Val(:a)) == ca.a

    @test Base.to_indices(ca, (:a, :b)) == (:a, :b)
    @test Base.to_indices(ca, (1, 2)) == (1, 2)
    @test Base.to_index(ca, :a) == :a

    #OffsetArray stuff
    part_ax = PartitionedAxis(2, Axis(a = 1, b = 2))
    oaca = ComponentArray(OffsetArray(collect(1:5), -1), Axis(a = 0, b = ViewAxis(1:4, part_ax)))
    temp_ca = ComponentArray(collect(1:5), Axis(a = 1, b = ViewAxis(2:5, part_ax)))
    @test oaca.a == temp_ca.a
    @test oaca.b[1].a == temp_ca.b[1].a
    @test oaca[0] == temp_ca[1]
    @test oaca[4] == temp_ca[5]
    @test axes(oaca) == axes(getdata(oaca))

    # Issue #56
    A = ComponentArray(rand(4, 10), Axis(a = 1:2, b = 3:4), FlatAxis())
    A_vec = A[:, 1]
    A_mat = A[:, 1:2]
    @test A_vec isa ComponentVector
    @test A_mat isa ComponentMatrix
    @test getdata(A_vec) isa Vector
    @test getdata(A_mat) isa Matrix

    # Issue #70
    let
        ca = ComponentVector(a = 1, b = 2, c = 3)
        @test_throws BoundsError ca[:a, :b]
    end

    # Issue # 87: Conversion/promotion
    let
        ax1 = Axis((; x1 = 1))
        ax2 = Axis((; x2 = 1))
        A1 = ComponentMatrix(zeros(1, 1), ax1, ax1)
        A2 = ComponentMatrix(zeros(1, 1), ax2, ax2)
        A = [A for A in [A1, A2]]
        @test A[1] == A1
        @test A[2] == A2
    end

    # Issue # 94: No getindex pirates
    @test_throws BoundsError a[]

    # Issue #112: InvertedIndices
    @test ca[Not(3)] == getdata(ca)[Not(3)]
    @test ca[Not(2:3)] == getdata(ca)[Not(2:3)]

    # Issue #248: Indexing ComponentMatrix with FlatAxis components
    @test cmat3[:a, :a] == cmat3check[1, 1]
    @test cmat3[:a, :b] == cmat3check[1, 2:5]
    @test cmat3[:a, :c] == reshape(cmat3check[1, 6:11], 3, 2)
    @test cmat3[:b, :a] == cmat3check[2:5, 1]
    @test cmat3[:b, :b] == cmat3check[2:5, 2:5]
    @test cmat3[:b, :c] == reshape(cmat3check[2:5, 6:11], 4, 3, 2)
    @test cmat3[:c, :a] == reshape(cmat3check[6:11, 1], 3, 2)
    @test cmat3[:c, :b] == reshape(cmat3check[6:11, 2:5], 3, 2, 4)
    @test cmat3[:c, :c] == reshape(cmat3check[6:11, 6:11], 3, 2, 3, 2)

    # https://discourse.julialang.org/t/no-method-error-reshape-when-solving-ode-with-componentarrays-jl/126342
    x = ComponentVector(x = 1.0, y = 0.0, z = 0.0)
    @test reshape(x, axes(x)...) === x
    @test reshape(x, axes(x)) === x
    @test reshape(a, axes(ca)...) isa Vector{Float64}

    # Issue #265: Multi-symbol indexing with matrix components
    @test ca2.c[[:a, :b]].b isa AbstractMatrix
end

@testset "Set" begin
    temp = deepcopy(ca2)
    tempmat = deepcopy(cmat2)

    temp.c.a .= 1000

    view(view(tempmat, :b, :b)[1, 1], :a, :a)[:a, :a] = 100000
    @view(tempmat[:b, :a])[2].b = 1000

    @test temp.c.a.a == 1000

    @test tempmat["b", "b"][1, 1]["a", :a][:a, :a] == 100000
    @test tempmat[:b, :a][2].b == 1000

    temp_b = deepcopy(temp.b)
    temp.b .= temp.b .* 100
    @test temp.b[1] == temp_b[1] .* 100

    temp2 = deepcopy(ca)
    temp3 = deepcopy(ca_MVector)
    @test (temp2 .= ca .* 1) isa ComponentArray
    @test (temp2 .= temp2 .* a .+ 1) isa typeof(temp2)
    @test (temp2 .= ca .* ca_SVector) isa typeof(temp2)
    @test (temp3 .= ca .* ca_SVector) isa typeof(temp3)

    temp2.b = ca.b .+ 1
    @test temp2.b == ca.b .+ 1

    setproperty!(temp2, :a, 20)
    @test temp2.a == 20

    setproperty!(temp2, Val(:b), zeros(2))
    @test temp2.b == zeros(2)

    tempmat .= 0
    @test tempmat[:b, :a][2].b == 0

    temp = deepcopy(cmat)
    @test all((temp[:c, :c][:a, :a] .= 0) .== 0)

    A = ComponentArray(zeros(Int, 4, 4), Axis(x = r2v(1:4)), Axis(x = r2v(1:4)))
    A[1, :] .= 1
    @test A[1, :] == ComponentVector(x = ones(Int, 4))
end

@testset "Properties" begin
    @test hasproperty(ca2, :a) # ComponentArray
    @test hasproperty(ca2.b, :a) # LazyArray

    @test propertynames(ca2) == (:a, :b, :c) # ComponentArray
    @test propertynames(ca2.b) == (:a, :b) # LazyArray

    @test haskey(ca2, :a) # ComponentArray
    @test haskey(ca2.b, 1) # LazyArray

    @test keys(ca2) == (:a, :b, :c)
    @test keys(ca2.b) == Base.OneTo(2)
end

@testset "Component Index" begin
    let
        ca = ComponentArray(a = 1, b = 2, c = [3, 4], d = (a = [5, 6, 7], b = 8))
        cmat = ca * ca'

        cidx = reshape((1:(2 * 3)) .+ 2, 2, 3)
        ca2 = ComponentArray(a = 1, b = 2, c = cidx, d = (a = [9, 10, 11], b = 12))

        @testset "ComponentIndex" begin
            ax = getaxes(ca)[1]
            @test ax[:a] == ax[1] ==
                  ComponentArrays.ComponentIndex(1, ComponentArrays.NullAxis())
            @test ax[:c] == ax[3:4] ==
                  ComponentArrays.ComponentIndex(3:4, ShapedAxis(size(3:4)))
            @test ax[:d] == ComponentArrays.ComponentIndex(5:8, Axis(a = r2v(1:3), b = 4))
            @test ax[(:a, :c)] == ax[[:a, :c]] ==
                  ComponentArrays.ComponentIndex([1, 3, 4], Axis(a = 1, c = r2v(2:3)))
            ax2 = getaxes(ca2)[1]
            @test ax2[(:a, :c)] == ax2[[:a, :c]] ==
                  ComponentArrays.ComponentIndex(
                      [1, 3:8...], Axis(a = 1, c = ViewAxis(2:7, ShapedAxis((2, 3)))))

            @test length(ComponentArrays.ComponentIndex(1, ComponentArrays.NullAxis())) == 1
            @test length(ComponentArrays.ComponentIndex(3:4, ShapedAxis(size(3:4)))) == 2
            @test length(ComponentArrays.ComponentIndex(5:8, Axis(a = r2v(1:3), b = 4))) ==
                  4
            @test length(ComponentArrays.ComponentIndex([1, 3, 4], Axis(a = 1, c = r2v(2:3)))) ==
                  3
            @test length(ComponentArrays.ComponentIndex(
                [1, 3:8...], Axis(a = 1, c = ViewAxis(2:7, ShapedAxis((2, 3)))))) == 7
        end

        @testset "KeepIndex" begin
            @test ca[KeepIndex(:a)] == ca[KeepIndex(1)] == ComponentArray(a = 1)
            @test ca[KeepIndex(:b)] == ca[KeepIndex(2)] == ComponentArray(b = 2)
            @test ca[KeepIndex(:c)] == ca[KeepIndex(3:4)] == ComponentArray(c = [3, 4])
            @test ca[KeepIndex(:d)] == ca[KeepIndex(5:8)] ==
                  ComponentArray(d = (a = [5, 6, 7], b = 8))

            @test ca[KeepIndex(1:2)] == ComponentArray(a = 1, b = 2)
            @test ca[KeepIndex(1:3)] == ComponentArray([1, 2, 3], Axis(a = 1, b = 2)) # Drops c axis
            @test ca[KeepIndex(2:5)] ==
                  ComponentArray([2, 3, 4, 5], Axis(b = 1, c = r2v(2:3)))
            @test ca[KeepIndex(3:end)] ==
                  ComponentArray(c = [3, 4], d = (a = [5, 6, 7], b = 8))

            @test ca[KeepIndex(:)] == ca

            @test cmat[KeepIndex(:a), KeepIndex(:b)] ==
                  ComponentArray(fill(2, 1, 1), Axis(a = 1), Axis(b = 1))
            @test cmat[KeepIndex(:), KeepIndex(:c)] ==
                  ComponentArray((1:8) * (3:4)', getaxes(ca)[1], Axis(c = r2v(1:2)))
            @test cmat[KeepIndex(2:5), 1:2] ==
                  ComponentArray((2:5) * (1:2)', Axis(b = 1, c = r2v(2:3)), ShapedAxis(size(1:2)))
            @test cmat[KeepIndex(2), KeepIndex(3)] ==
                  ComponentArray(fill(2 * 3, 1, 1), Axis(b = 1), FlatAxis())
            @test cmat[KeepIndex(2), 3] == ComponentArray(b = 2 * 3)
        end
    end
end

@testset "Similar" begin
    @test similar(ca) isa typeof(ca)
    @test similar(ca2) isa typeof(ca2)
    @test similar(ca, Float32) isa typeof(ca_Float32)
    @test eltype(similar(ca, ForwardDiff.Dual)) == ForwardDiff.Dual
    @test similar(ca, 5) isa typeof(getdata(ca))
    @test similar(ca, Float32, 5) isa typeof(getdata(ca_Float32))
    @test similar(cmat, 5, 5) isa typeof(getdata(cmat))

    # Issue #206
    x = ComponentArray(a = false, b = true)
    @test typeof(x) == typeof(zero(x))
end

@testset "Copy" begin
    @test copy(ca) == ca
    @test deepcopy(ca) == ca
end

@testset "Convert" begin
    @test NamedTuple(ca) == nt
    @test NamedTuple(ca.c) == c
    @test convert(typeof(ca), a) == ca
    @test convert(typeof(ca), ca) == ca
    @test convert(typeof(cmat), cmat) == cmat

    @test convert(Array, ca) == getdata(ca)
    @test convert(Matrix{Float32}, cmat) isa Matrix{Float32}

    tr = Tracker.param(ca)
    ca_ = convert(typeof(ca), tr)
    @test ca_.a == ca.a
end

@testset "Broadcasting" begin
    temp = deepcopy(ca)
    @test eltype(Float32.(ca)) == Float32
    @test ca .* ca' == cmat
    @test 1 .* (ca .+ ca) == ComponentArray(a .+ a, getaxes(ca))
    @test typeof(ca .+ cmat) == typeof(cmat)
    @test getaxes(false .* ca .* ca') == (ax, ax)
    @test getaxes(false .* ca' .* ca) == (ax, ax)
    @test (vec(temp) .= vec(ca_Float32)) isa ComponentArray

    @test_broken getdata(ca_MVector .* ca_MVector) isa MArray
    @test_broken typeof(ca .* ca_MVector) == typeof(ca)
    @test_broken typeof(ca_SVector .* ca) == typeof(ca)
    @test_broken typeof(ca_SVector .* ca_SVector) == typeof(ca_SVector)
    @test_broken typeof(ca_SVector .* ca_MVector) == typeof(ca_SVector)
    @test_broken typeof(ca_SVector' .+ ca) == typeof(cmat)
    @test_broken getdata(ca_SVector' .+ ca_SVector') isa StaticArrays.StaticArray
    @test_broken getdata(ca_SVector .* ca_SVector') isa StaticArrays.StaticArray
    @test_broken ca_SVector .* ca .+ a .- 1 isa ComponentArray

    # Issue #31 (with Complex as a stand-in for Dual)
    @test reshape(Complex.(ca, Float32.(a)), size(ca)) isa ComponentArray{Complex{Float64}}

    # Issue #34 : Different Axis types
    x1 = ComponentArray(a = [1.1, 2.1], b = [0.1])
    x2 = ComponentArray(a = [1.1, 2.1], b = 0.1)
    x3 = ComponentArray(a = [1.1, 2.1], c = [0.1])
    xmat = x1 .* x2'
    x1mat = x1 .* x1'
    @test x1 + x2 isa Vector
    @test x1 + x3 isa Vector
    @test x2 + x3 isa Vector
    @test x1 .* x2 isa Vector
    @test xmat + x1mat isa ComponentArray
    @test xmat isa ComponentArray
    @test getaxes(xmat) == (getaxes(x1)[1], getaxes(x2)[1])
    @test getaxes(x1mat + xmat) == (getaxes(x1)[1], FlatAxis())
    @test getaxes(x1mat + xmat') == (FlatAxis(), getaxes(x1)[1])

    @test map(sqrt, ca) isa ComponentArray
    @test map(+, ca, sqrt.(ca)) isa ComponentArray
    @test map(+, sqrt.(ca), Float32.(ca), ca) isa ComponentArray
    @test map(+, ca, getdata(ca)) isa Array
    @test map(+, ca, ComponentArray(v = getdata(ca))) isa Array

    x1 .+= x2
    @test getdata(x1) == 2getdata(x2)

    # Issue #60
    x4 = ComponentArray(rand(3, 3), Axis(x = 1, y = 2, z = 3), Axis(x = 1, y = 2, z = 3))
    @test x4 + I(3) isa ComponentMatrix

    # Issue #98
    let
        x = ComponentArray(x = 1:3)
        y = ComponentArray(y = 1:3)
        z = ComponentArray(z = 1:3)
        yz = y * z'
        @test yz * x == ComponentArray(y = [14, 28, 42])
        @test getdata(yz) * x == [14, 28, 42]
        @test x .+ y .+ z isa Vector
        @test Complex.(x, y) isa Vector
        @test Complex.(x, x) isa ComponentVector
        @test Complex.(x, y') isa ComponentMatrix
    end
end

@testset "Math" begin
    a_t = collect(a')

    @test ca * ca' == collect(cmat)
    @test ca * ca' == a * a'
    @test ca' * ca == a' * a
    @test cmat * ca == ComponentArray(cmat * a, getaxes(ca))
    @test cmat' * ca isa AbstractArray
    @test a' * ca isa Number
    @test cmat'' == cmat
    @test ca'' == ca
    @test ca.c' * cmat[:c, :c] * ca.c isa Number
    @test ca * 1 isa ComponentVector
    @test size(ca' * 1) == size(ca')
    @test a' * ca isa Number
    @test a_t * ca isa AbstractArray
    @test a' * cmat isa Adjoint
    @test a_t * cmat isa AbstractArray
    @test cmat * ca isa AbstractVector
    @test ca + ca + ca isa typeof(ca)
    @test a + ca + ca isa typeof(ca)
    @test a * ca' isa AbstractMatrix

    @test ca * transpose(ca) == collect(cmat)
    @test ca * transpose(ca) == a * transpose(a)
    @test transpose(ca) * ca == transpose(a) * a
    @test ca' * cmat == ComponentArray(a' * getdata(cmat), getaxes(ca))
    @test transpose(transpose(cmat)) == cmat
    @test transpose(transpose(ca)) == ca
    @test transpose(ca.c) * cmat[:c, :c] * ca.c isa Number
    @test size(transpose(ca) * 1) == size(transpose(ca))
    @test transpose(a) * ca isa Number
    @test transpose(a) * cmat isa Transpose
    @test a * transpose(ca) isa AbstractMatrix

    temp = deepcopy(ca)
    temp .= (cmat + I) \ ca
    @test temp isa ComponentArray
    @test (ca' / (cmat' + I))' == (cmat + I) \ ca
    @test cmat * ((cmat + I) \ ca) isa AbstractArray
    @test inv(cmat + I) isa AbstractArray

    tempmat = deepcopy(cmat)

    @test ldiv!(temp, lu(cmat + I), ca) isa ComponentVector
    @test ldiv!(getdata(temp), lu(cmat + I), ca) isa AbstractVector
    @test ldiv!(tempmat, lu(cmat + I), cmat) isa ComponentMatrix
    @test ldiv!(getdata(tempmat), lu(cmat + I), cmat) isa AbstractMatrix

    c = (a = 2, b = [1, 2]);
    x = ComponentArray(
        a = 5, b = [
            (a = 20.0, b = 3.0), (a = 33.0, b = 2.0), (a = 44.0, b = 3.0)], c = c)
    @test ldiv!(rand(10), Diagonal(x), x) isa Vector

    vca2 = vcat(ca2', ca2')
    hca2 = hcat(ca2, ca2)
    temp = ComponentVector(q = 100, r = rand(3, 3, 3))
    vtempca = [temp; ca]
    @test all(vca2[1, :] .== ca2)
    @test all(hca2[:, 1] .== ca2)
    @test all(vca2' .== hca2)
    @test hca2[:a, :] == vca2[:, :a]
    @test vtempca isa ComponentVector
    @test vtempca.r == temp.r
    @test vtempca.c == ca.c
    @test length(vtempca) == length(temp) + length(ca)
    @test [ca; ca; ca] isa Vector
    @test vcat(ca, 100) isa Vector
    @test [ca' ca']' isa Vector
    @test keys(getaxes([ca' temp']')[1]) == (:a, :b, :c, :q, :r)

    # Getting serious about axes
    let
        ab = ComponentArray(a = 1, b = 5)
        cd = ComponentArray(c = 3, d = 7)
        ab_ab = ab * ab'
        ab_cd = ab * cd' + I
        cd_ab = cd * ab'
        cd_cd = cd * cd'
        AB = Axis(a = 1, b = 2)
        CD = Axis(c = 1, d = 2)
        _AB = Axis(a = 2, b = 3)
        _CD = Axis(c = 2, d = 3)
        ABCD = Axis(a = 1, b = 2, c = 3, d = 4)
        CDAB = Axis(c = 1, d = 2, a = 3, b = 4)

        # Cats
        @test [ab_ab; ab_ab] isa Matrix
        @test [ab_ab; ab_cd] isa Matrix
        @test getaxes([ab_ab; cd_ab]) == (ABCD, AB)
        @test getaxes([ab_ab ab_cd]) == (AB, ABCD)
        @test getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
        @test getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
        @test getaxes([ab ab_cd]) == (AB, _CD)
        @test getaxes([ab_cd ab]) == (AB, CD)
        @test getaxes([ab'; cd_ab]) == (_CD, AB)
        @test getaxes([cd'; cd_ab']) == (_AB, CD)
        @test getaxes([cd'; cd_ab']) == (_AB, CD)

        # Math
        @test getaxes(ab_cd * cd) == (AB,)
        @test getaxes(cd_ab' * cd) == (AB,)
        @test getaxes(cd' * cd_ab) == (FlatAxis(), AB)
        @test getaxes(cd' * cd_ab') == (FlatAxis(), CD)
        @test getaxes(cd_ab' * cd_ab) == (AB, AB)
        @test getaxes(cd_ab' * ab_cd') == (AB, AB)
        @test getaxes(ab_cd * ab_cd') == (AB, AB)
        @test getaxes(ab_cd \ ab) == (CD,)
        @test getaxes(ab_cd' \ cd) == (AB,)
        @test getaxes(cd' / ab_cd) == (FlatAxis(), AB)
        @test getaxes(ab' / ab_cd') == (FlatAxis(), CD)
        @test getaxes(ab_cd \ ab_cd) == (CD, CD)
    end

    # Issue #33
    smat = @SMatrix [1 2; 3 4]
    b = ComponentArray(a = 1, b = 2)
    @test smat * b isa StaticArray

    # Issue #86: Matrix multiplication
    in1 = ComponentArray(u1 = 1)
    in2 = ComponentArray(u2 = 1)
    out1 = ComponentArray(y1 = 1)
    out2 = ComponentArray(y2 = 1)
    s1_D = out1 * in1'
    s2_D = out2 * in2'
    @test getaxes(s1_D * s2_D) == (Axis(y1 = 1), Axis(u2 = 1))
    @test getaxes(s2_D * s1_D) == (Axis(y2 = 1), Axis(u1 = 1))
    @test getaxes((s1_D * s2_D) * in2) == getaxes(s1_D * (s2_D * in2)) == (Axis(y1 = 1),)
    @test getaxes((s2_D * s1_D) * in1) == getaxes(s2_D * (s1_D * in1)) == (Axis(y2 = 1),)
    @test getaxes(out1' * (s1_D * s2_D)) == getaxes(transpose(out1) * (s1_D * s2_D)) ==
          (FlatAxis(), Axis(u2 = 1))

    @test ComponentArrays.ArrayInterface.lu_instance(cmat).factors isa ComponentMatrix
    @test ComponentArrays.ArrayInterface.parent_type(cmat) === Matrix{Float64}
end

@testset "Static Unpack" begin
    x = ComponentArray(a = 5, b = [4, 1], c = [1 2; 3 4], d = (e = 2, f = [6, 30.0]))
    @static_unpack a, b, c, d = x
    @static_unpack e, f = x.d .+ 0

    @test a isa Float64
    @test b isa SVector{2, Float64}
    @test c isa SMatrix{2, 2, Float64, 4}
    @test d isa ComponentArray
    @test e isa Float64
    @test f isa SVector{2, Float64}

    @static_unpack a = x
    @static_unpack (; b, c) = x

    @test a isa Float64
    @test b isa SVector{2, Float64}
    @test c isa SMatrix{2, 2, Float64, 4}
end

@testset "Plot Utilities" begin
    lab = labels(ca2)
    @test lab == [
        "a",
        "b[1].a.a",
        "b[1].a.b",
        "b[1].b",
        "b[2].a.a",
        "b[2].a.b",
        "b[2].b",
        "c.a.a",
        "c.a.b[1]",
        "c.a.b[2]",
        "c.b[1,1]",
        "c.b[2,1]",
        "c.b[1,2]",
        "c.b[2,2]"
    ]
    @test label2index(ca2, "c.b") == collect(11:14)

    # Issue #74
    lab2 = labels(ComponentArray(
        a = 1, aa = ones(2), ab = [(a = 1, aa = ones(2)), (a = 1, aa = ones(2))],
        ac = (a = 1, ab = ones(2, 2))))
    @test label2index(lab2, "a") == [1]
    @test label2index(lab2, "aa") == collect(2:3)
    @test label2index(lab2, "ab") == collect(4:9)
    @test label2index(lab2, "ab[1].aa") == collect(5:6)
    @test label2index(lab2, "ac") == collect(10:14)
    @test label2index(lab2, "ac.a") == [10]
    @test label2index(lab2, "ac.ab") == collect(11:14)
end

@testset "Uncategorized Issues" begin
    # Issue #25
    @test sum(abs2, cmat) == sum(abs2, getdata(cmat))

    # Issue #40
    r0 = [1131.340, -2282.343, 6672.423]u"km"
    v0 = [-5.64305, 4.30333, 2.42879]u"km/s"
    rv0 = ComponentArray(r = r0, v = v0)
    zrv0 = zero(rv0)
    @test all(zero(cmat) * ca .== zero(ca))
    @test typeof(zrv0) === typeof(rv0)
    @test typeof(zrv0.r[1]) == typeof(rv0[1])

    # Issue #140
    @test ComponentArrays.ArrayInterface.indices_do_not_alias(typeof(ca)) == true
    @test ComponentArrays.ArrayInterface.instances_do_not_alias(typeof(ca)) == false

    # Issue #193
    # Make sure we aren't doing type piracy on `reshape`
    @test ndims(dropdims(ones(1, 1), dims = (1, 2))) == 0
    @test reshape([1]) == fill(1, ())

    # Tests for stack function (introduced in Julia 1.9, always available in Julia 1.10+)
    # `stack` was introduced in Julia 1.9
    # Issue #254
    x = ComponentVector(a = [1, 2])
    y = ComponentVector(a = [3, 4])
    xy = stack([x, y])
    # The data in `xy` should be the same as what we'd get if we used plain Vectors:
    @test getdata(xy) == stack(getdata.([x, y]))
    # Check the axes.
    xy_ax = getaxes(xy)
    # Should have two axes since xy should be a ComponentMatrix.
    @test length(xy_ax) == 2
    # First axis should be the same as x.
    @test xy_ax[1] == only(getaxes(x))
    # Second axis should be a FlatAxis.
    @test xy_ax[2] == FlatAxis()

    # Does the dims argument to stack work?
    # Using `dims=2` should be the same as the default value.
    xy2 = stack([x, y]; dims = 2)
    @test xy2 == xy
    # Using `dims=1` should stack things vertically.
    xy3 = stack([x, y]; dims = 1)
    @test all(xy3[1, :a] .== xy[:a, 1])
    @test all(xy3[2, :a] .== xy[:a, 2])

    # But can we stack 2D arrays?
    x = ComponentVector(a = [1, 2])
    y = ComponentVector(b = [3, 4])
    X = x .* y'
    Y = x .* y' .+ 4
    XY = stack([X, Y])
    # The data in `XY` should be the same as what we'd get if we used plain Vectors:
    @test getdata(XY) == stack(getdata.([X, Y]))
    # Check the axes.
    XY_ax = getaxes(XY)
    # Should have three axes since XY should be a 3D ComponentArray.
    @test length(XY_ax) == 3
    # First two axes should be the same as XY.
    @test XY_ax[1] == getaxes(XY)[1]
    @test XY_ax[2] == getaxes(XY)[2]
    # Third should be a FlatAxis.
    @test XY_ax[3] == FlatAxis()
    # Should test indexing too.
    @test all(XY[:a, :b, 1] .== X)
    @test all(XY[:a, :b, 2] .== Y)

    # Make sure the dims argument works.
    # Using `dims=3` should be the same as the default value.
    XY_d3 = stack([X, Y]; dims = 3)
    @test XY_d3 == XY
    # Using `dims=2` stacks along the second axis.
    XY_d2 = stack([X, Y]; dims = 2)
    @test all(XY_d2[:a, 1, :b] .== XY[:a, :b, 1])
    @test all(XY_d2[:a, 2, :b] .== XY[:a, :b, 2])
    # Using `dims=1` stacks along the first axis.
    XY_d1 = stack([X, Y]; dims = 1)
    @test all(XY_d1[1, :a, :b] .== XY[:a, :b, 1])
    @test all(XY_d1[2, :a, :b] .== XY[:a, :b, 2])

    # Issue #254, tuple of arrays:
    x = ComponentVector(a = [1, 2])
    y = ComponentVector(b = [3, 4])
    Xstack1 = stack((x, y, x); dims = 1)
    Xstack1_noca = stack((getdata(x), getdata(y), getdata(x)); dims = 1)
    @test all(Xstack1 .== Xstack1_noca)
    @test all(Xstack1[1, :a] .== Xstack1_noca[1, :])
    @test all(Xstack1[2, :a] .== Xstack1_noca[2, :])

    # Issue #254, Array of tuples.
    Xstack2 = stack(ComponentArray(a = (1, 2, 3), b = (4, 5, 6)))
    Xstack2_noca = stack([(1, 2, 3), (4, 5, 6)])
    @test all(Xstack2 .== Xstack2_noca)
    @test all(Xstack2[:, :a] .== Xstack2_noca[:, 1])
    @test all(Xstack2[:, :b] .== Xstack2_noca[:, 2])

    Xstack2_d1 = stack(ComponentArray(a = (1, 2, 3), b = (4, 5, 6)); dims = 1)
    Xstack2_noca_d1 = stack([(1, 2, 3), (4, 5, 6)]; dims = 1)
    @test all(Xstack2_d1 .== Xstack2_noca_d1)
    @test all(Xstack2_d1[:a, :] .== Xstack2_noca_d1[1, :])
    @test all(Xstack2_d1[:b, :] .== Xstack2_noca_d1[2, :])

    # Issue #254, generator of arrays.
    Xstack3 = stack(ComponentArray(z = [x, x]) for x in 1:4)
    Xstack3_noca = stack([x, x] for x in 1:4)
    # That should give me
    # [1 2 3 4;
    #  1 2 3 4]
    @test all(Xstack3 .== Xstack3_noca)
    @test all(Xstack3[:z, 1] .== Xstack3_noca[:, 1])
    @test all(Xstack3[:z, 2] .== Xstack3_noca[:, 2])
    @test all(Xstack3[:z, 3] .== Xstack3_noca[:, 3])
    @test all(Xstack3[:z, 4] .== Xstack3_noca[:, 4])

    Xstack3_d1 = stack(ComponentArray(z = [x, x]) for x in 1:4; dims = 1)
    Xstack3_noca_d1 = stack([x, x] for x in 1:4; dims = 1)
    # That should give me
    # [1 1;
    #  2 2;
    #  3 3;
    #  4 4;]
    @test all(Xstack3_d1 .== Xstack3_noca_d1)
    @test all(Xstack3_d1[1, :z] .== Xstack3_noca_d1[1, :])
    @test all(Xstack3_d1[2, :z] .== Xstack3_noca_d1[2, :])
    @test all(Xstack3_d1[3, :z] .== Xstack3_noca_d1[3, :])
    @test all(Xstack3_d1[4, :z] .== Xstack3_noca_d1[4, :])

    # Issue #254, map then stack.
    Xstack4_d1 = stack(x -> ComponentArray(a = x, b = [x+1, x+2]), [5 6; 7 8]; dims = 1)  # map then stack
    Xstack4_noca_d1 = stack(x -> [x, x+1, x+2], [5 6; 7 8]; dims = 1)  # map then stack
    @test all(Xstack4_d1 .== Xstack4_noca_d1)
    @test all(Xstack4_d1[:, :a] .== Xstack4_noca_d1[:, 1])
    @test all(Xstack4_d1[:, :b] .== Xstack4_noca_d1[:, 2:3])

    Xstack4_d2 = stack(x -> ComponentArray(a = x, b = [x+1, x+2]), [5 6; 7 8]; dims = 2)  # map then stack
    Xstack4_noca_d2 = stack(x -> [x, x+1, x+2], [5 6; 7 8]; dims = 2)  # map then stack
    @test all(Xstack4_d2 .== Xstack4_noca_d2)
    @test all(Xstack4_d2[:a, :] .== Xstack4_noca_d2[1, :])
    @test all(Xstack4_d2[:b, :] .== Xstack4_noca_d2[2:3, :])

    Xstack4_dcolon = stack(x -> ComponentArray(a = x, b = [x+1, x+2]), [5 6; 7 8]; dims = :)  # map then stack
    Xstack4_noca_dcolon = stack(x -> [x, x+1, x+2], [5 6; 7 8]; dims = :)  # map then stack
    @test all(Xstack4_dcolon .== Xstack4_noca_dcolon)
    @test all(Xstack4_dcolon[:a, :, :] .== Xstack4_noca_dcolon[1, :, :])
    @test all(Xstack4_dcolon[:b, :, :] .== Xstack4_noca_dcolon[2:3, :, :])

    # Test that we maintain higher-order components during vcat.
    x = ComponentVector(a = rand(Float64, 2, 3, 4), b = rand(Float64, 4, 3, 2))
    y = ComponentVector(c = rand(Float64, 3, 4, 2), d = rand(Float64, 3, 2, 4))
    xy = vcat(x, y)
    @test size(xy.a) == size(x.a)
    @test size(xy.b) == size(x.b)
    @test size(xy.c) == size(y.c)
    @test size(xy.d) == size(y.d)
    @test all(xy.a .≈ x.a)
    @test all(xy.b .≈ x.b)
    @test all(xy.c .≈ y.c)
    @test all(xy.d .≈ y.d)

    # Test fix https://github.com/Deltares/Ribasim/issues/2028
    a = range(0.0, 1.0, length = 0) |> collect
    b = range(0.0, 1.0; length = 2) |> collect
    c = range(0.0, 1.0, length = 3) |> collect
    d = range(0.0, 1.0; length = 0) |> collect
    u = ComponentVector(a = a, b = b, c = c, d = d)

    function get_state_index(
            idx::Int,
            ::ComponentVector{A, B, <:Tuple{<:Axis{NT}}},
            component_name::Symbol
    ) where {A, B, NT}
        for (comp, range) in pairs(NT)
            if comp == component_name
                return range[idx]
            end
        end
        return nothing
    end

    @test_throws BoundsError get_state_index(1, u, :a)
    @test_throws BoundsError get_state_index(2, u, :a)
    @test get_state_index(1, u, :b) == 1
    @test get_state_index(2, u, :b) == 2
    @test get_state_index(1, u, :c) == 3
    @test get_state_index(2, u, :c) == 4
    @test get_state_index(3, u, :c) == 5
    @test_throws BoundsError get_state_index(1, u, :d)
    @test_throws BoundsError get_state_index(2, u, :d)

    # Must be a better way to make sure we can `Base.iterate` the `ViewAxis{UnitRange, Shaped1DAxis}`.
    nt = ComponentArrays.indexmap(getaxes(u)[1])
    for (i, idx) in enumerate(nt.a)
    end
    for (i, idx) in enumerate(nt.b)
        @test idx == i
    end
    for (i, idx) in enumerate(nt.c)
        @test idx == i + 2
    end
    for (i, idx) in enumerate(nt.d)
    end
end

@testset "axpy! / axpby!" begin
    y = ComponentArray(a = rand(4), b = rand(4))
    x = ComponentArray(a = rand(4), b = rand(4))
    ydata = copy(getdata(y))

    axpy!(2, x, y)
    @test getdata(y) == 2 .* getdata(x) .+ ydata

    x = ComponentArray(a = rand(4), c = rand(4))
    @test_throws ArgumentError axpy!(2, x, y)

    y = ComponentArray(a = rand(4), b = rand(4))
    x = ComponentArray(a = rand(4), b = rand(4))
    ydata = copy(getdata(y))

    axpby!(2, x, 3, y)
    @test getdata(y) == 2 .* getdata(x) .+ 3 .* ydata

    x = ComponentArray(a = rand(4), c = rand(4))
    @test_throws ArgumentError axpby!(2, x, 3, y)
end

@testset "Empty NamedTuple" begin
    @test ComponentArray(NamedTuple()) isa ComponentVector{Float32}
end

@testset "Functors" begin
    for carray in (ca, ca_Float32, ca_MVector, ca_SVector, ca_composed, ca2, caa)
        θ, re = Functors.functor(carray)
        @test θ isa NamedTuple
        @test re(θ) == carray
    end
end

@testset "Autodiff" begin
    include("autodiff_tests.jl")
end

@testset "GPU" begin
    include("gpu_tests.jl")
end

@testset "Reactant" begin
    include("reactant_tests.jl")
end

@testset "JET" begin
    include("jet_tests.jl")
end
