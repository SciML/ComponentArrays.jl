include("shared/test_setup.jl")

# Issue #25
@test sum(abs2, cmat) == sum(abs2, getdata(cmat))

# Issue #40
r0 = [1131.34, -2282.343, 6672.423]u"km"
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
Xstack4_d1 = stack(x -> ComponentArray(a = x, b = [x + 1, x + 2]), [5 6; 7 8]; dims = 1)  # map then stack
Xstack4_noca_d1 = stack(x -> [x, x + 1, x + 2], [5 6; 7 8]; dims = 1)  # map then stack
@test all(Xstack4_d1 .== Xstack4_noca_d1)
@test all(Xstack4_d1[:, :a] .== Xstack4_noca_d1[:, 1])
@test all(Xstack4_d1[:, :b] .== Xstack4_noca_d1[:, 2:3])

Xstack4_d2 = stack(x -> ComponentArray(a = x, b = [x + 1, x + 2]), [5 6; 7 8]; dims = 2)  # map then stack
Xstack4_noca_d2 = stack(x -> [x, x + 1, x + 2], [5 6; 7 8]; dims = 2)  # map then stack
@test all(Xstack4_d2 .== Xstack4_noca_d2)
@test all(Xstack4_d2[:a, :] .== Xstack4_noca_d2[1, :])
@test all(Xstack4_d2[:b, :] .== Xstack4_noca_d2[2:3, :])

Xstack4_dcolon = stack(x -> ComponentArray(a = x, b = [x + 1, x + 2]), [5 6; 7 8]; dims = :)  # map then stack
Xstack4_noca_dcolon = stack(x -> [x, x + 1, x + 2], [5 6; 7 8]; dims = :)  # map then stack
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
u = ComponentVector(; a, b, c, d)

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
