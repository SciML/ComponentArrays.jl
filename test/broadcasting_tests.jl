include("shared/test_setup.jl")

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
