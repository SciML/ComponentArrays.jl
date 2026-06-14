include("shared/test_setup.jl")

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
