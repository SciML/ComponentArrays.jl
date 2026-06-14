include("shared/test_setup.jl")

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
