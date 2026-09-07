include("shared/test_setup.jl")

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
