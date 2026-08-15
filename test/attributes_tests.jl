include("shared/test_setup.jl")

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

@testset "AbstractAxis generic interface" begin
    @eval module ExternalAxisInterface
    using ComponentArrays: AbstractAxis

    struct TwoComponentAxis <: AbstractAxis{(left = 1, right = 2)} end
    end

    axis = ExternalAxisInterface.TwoComponentAxis()
    @test keys(axis) == (:left, :right)
    @test axis[:left] == axis[Val(:left)]
    @test length(axis[:left]) == 1
    @test length(axis[(:left, :right)]) == 2
    @test firstindex(axis) == 1
    @test lastindex(axis) == 2

    x = ComponentArray([10, 20], axis)
    @test x isa ComponentArray
    @test getdata(x) == [10, 20]
    @test getaxes(x) == (axis,)
end
