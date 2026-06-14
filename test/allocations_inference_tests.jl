include("shared/test_setup.jl")

@test @ballocated($ca.c.a.a) == 0
@test @ballocated(@view $ca[:c]) == 0
@test @ballocated(@view $cmat[:c, :c]) == 0

f = (out, x) -> (out .= x .+ x)
out = deepcopy(ca)
@test @ballocated($f($out, $ca)) == 0
