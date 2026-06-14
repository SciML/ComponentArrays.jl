include("shared/test_setup.jl")

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

c = (a = 2, b = [1, 2])
x = ComponentArray(;
    a = 5,
    b = [(a = 20.0, b = 3.0), (a = 33.0, b = 2.0), (a = 44.0, b = 3.0)],
    c,
)
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
    # These tests fail on Julia 1.13+ due to changed hvcat dispatch behavior
    # The ComponentArrays.hvcat method is not being selected over LinearAlgebra's
    if VERSION < v"1.13.0-"
        @test getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
        @test getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
    else
        @test_broken getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
        @test_broken getaxes([ab_ab ab_cd; cd_ab cd_cd]) == (ABCD, ABCD)
    end
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
