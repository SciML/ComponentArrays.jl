include("shared/test_setup.jl")

@test copy(ca) == ca
@test deepcopy(ca) == ca
