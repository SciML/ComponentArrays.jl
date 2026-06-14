using SafeTestsets

@safetestset "JET" begin
    include("jet_tests.jl")
end

@safetestset "Aqua" begin
    include("aqua_tests.jl")
end
