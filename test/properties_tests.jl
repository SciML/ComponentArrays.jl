include("shared/test_setup.jl")

@test hasproperty(ca2, :a) # ComponentArray
@test hasproperty(ca2.b, :a) # LazyArray

@test propertynames(ca2) == (:a, :b, :c) # ComponentArray
@test propertynames(ca2.b) == (:a, :b) # LazyArray

@test haskey(ca2, :a) # ComponentArray
@test haskey(ca2.b, 1) # LazyArray

@test keys(ca2) == (:a, :b, :c)
@test keys(ca2.b) == Base.OneTo(2)
