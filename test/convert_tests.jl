include("shared/test_setup.jl")

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
