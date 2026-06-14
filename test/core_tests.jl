using SafeTestsets

@safetestset "Allocations and Inference" begin
    include("allocations_inference_tests.jl")
end

@safetestset "Utilities" begin
    include("utilities_tests.jl")
end

@safetestset "Construction" begin
    include("construction_tests.jl")
end

@safetestset "Attributes" begin
    include("attributes_tests.jl")
end

@safetestset "Get" begin
    include("get_tests.jl")
end

@safetestset "Set" begin
    include("set_tests.jl")
end

@safetestset "Properties" begin
    include("properties_tests.jl")
end

@safetestset "Component Index" begin
    include("component_index_tests.jl")
end

@safetestset "Similar" begin
    include("similar_tests.jl")
end

@safetestset "Copy" begin
    include("copy_tests.jl")
end

@safetestset "Convert" begin
    include("convert_tests.jl")
end

@safetestset "Broadcasting" begin
    include("broadcasting_tests.jl")
end

@safetestset "Math" begin
    include("math_tests.jl")
end

@safetestset "Static Unpack" begin
    include("static_unpack_tests.jl")
end

@safetestset "Plot Utilities" begin
    include("plot_utilities_tests.jl")
end

@safetestset "Uncategorized Issues" begin
    include("uncategorized_issues_tests.jl")
end

@safetestset "axpy! / axpby!" begin
    include("axpy_axpby_tests.jl")
end

@safetestset "Empty NamedTuple" begin
    include("empty_namedtuple_tests.jl")
end

@safetestset "Functors" begin
    include("functors_tests.jl")
end
