using SciMLTesting

run_tests(;
    core = joinpath(@__DIR__, "core_tests.jl"),
    groups = Dict(
        "Autodiff" => (;
            env = joinpath(@__DIR__, "Autodiff"),
            body = joinpath(@__DIR__, "Autodiff", "autodiff_tests.jl"),
        ),
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = joinpath(@__DIR__, "GPU", "gpu_tests.jl"),
        ),
        "Downstream" => (;
            env = joinpath(@__DIR__, "Downstream"),
            body = joinpath(@__DIR__, "Downstream", "diffeq_tests.jl"),
        ),
        "Reactant" => (;
            env = joinpath(@__DIR__, "Reactant"),
            body = joinpath(@__DIR__, "Reactant", "reactant_tests.jl"),
        ),
        "nopre" => (;
            env = joinpath(@__DIR__, "nopre"),
            body = joinpath(@__DIR__, "nopre", "nopre_tests.jl"),
        ),
    ),
    all = ["Core"],
)
