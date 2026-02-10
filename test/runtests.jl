using Pkg
using Test

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env_dir)
    Pkg.activate(env_dir)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @time @testset "Core" begin
        include("core_tests.jl")
    end
elseif GROUP == "Autodiff"
    activate_env("autodiff")
    @time @testset "Autodiff" begin
        include("autodiff/autodiff_tests.jl")
    end
elseif GROUP == "GPU"
    activate_env("gpu")
    @time @testset "GPU" begin
        include("gpu/gpu_tests.jl")
    end
elseif GROUP == "Downstream"
    activate_env("downstream")
    @time @testset "Downstream" begin
        include("downstream/diffeq_tests.jl")
    end
elseif GROUP == "Reactant"
    activate_env("reactant")
    @time @testset "Reactant" begin
        include("reactant/reactant_tests.jl")
    end
elseif GROUP == "nopre"
    activate_env("nopre")
    @time @testset "JET" begin
        include("nopre/jet_tests.jl")
    end
    @time @testset "Aqua" begin
        include("nopre/aqua_tests.jl")
    end
else
    error("Unknown test group: $GROUP. Valid groups: All, Core, Autodiff, GPU, Downstream, Reactant, nopre")
end
