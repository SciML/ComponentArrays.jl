include("shared/test_setup.jl")

for carray in (ca, ca_Float32, ca_MVector, ca_SVector, ca_composed, ca2, caa)
    θ, re = Functors.functor(carray)
    @test θ isa NamedTuple
    @test re(θ) == carray
end
