using ComponentArrays
using Documenter
using Documenter.Remotes: GitHub

DocMeta.setdocmeta!(ComponentArrays, :DocTestSetup, :(using ComponentArrays))

makedocs(;
    modules = [ComponentArrays],
    format = Documenter.HTML(
        canonical = "https://sciml.github.io/ComponentArrays.jl/stable",
    ),
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Indexing Behavior" => "indexing_behavior.md",
        "Unpacking to StaticArrays" => "static_unpack.md",
        "Examples" => [
            "examples/DiffEqFlux.md",
            "examples/adaptive_control.md",
            "examples/ODE_jac.md",
            "examples/coulomb_control.md",
        ],
        "API" => "api.md",
    ],
    repo = GitHub("SciML/ComponentArrays.jl"),
    sitename = "ComponentArrays.jl",
    authors = "Jonnie Diegelman"
)

deploydocs(;
    repo = "github.com/SciML/ComponentArrays.jl.git",
)
