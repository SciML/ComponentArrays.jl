include("shared/test_setup.jl")

@test ComponentArray(NamedTuple()) isa ComponentVector{Float32}
