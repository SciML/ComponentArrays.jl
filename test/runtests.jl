if get(ENV, "GROUP", "") == "QA"
    qa_project = joinpath(@__DIR__, "qa")
    qa_file = joinpath(qa_project, "qa.jl")
    package_root = dirname(@__DIR__)
    source_url = "https://github.com/SciML/SciMLTesting.jl"
    source_rev = "a5cecca928f2c684c23505c3cd83921d71753e2e"
    # Julia 1.10 Pkg does not read [sources], so add the pinned source before instantiating.
    scimltesting = "PackageSpec(name=\"SciMLTesting\", url=$(repr(source_url)), rev=$(repr(source_rev)))"
    script = join(
        (
            "using Pkg",
            "Pkg.add($scimltesting)",
            "Pkg.develop(PackageSpec(path=$(repr(package_root))))",
            "Pkg.instantiate()",
            "include($(repr(qa_file)))",
        ), "; "
    )
    run(`$(Base.julia_cmd()) --project=$qa_project -e $script`)
else
    using SciMLTesting

    run_tests()
end
