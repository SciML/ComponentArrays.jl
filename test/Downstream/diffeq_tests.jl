using ComponentArrays
using DifferentialEquations
using OrdinaryDiffEqRosenbrock
using Sundials
using Test
using Unitful

@testset "Issue 31" begin
    function rober(vars, p, t)
        y₁, y₂, y₃ = vars
        k₁, k₂, k₃ = p
        D = similar(vars)
        D.y₁ = -k₁ * y₁ + k₃ * y₂ * y₃
        D.y₂ = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
        D.y₃ = k₂ * y₂^2
        return D
    end
    ic = ComponentArray(y₁ = 1.0, y₂ = 0.0, y₃ = 0.0)
    prob = ODEProblem(rober, ic, (0.0, 1.0e11), (0.04, 3.0e7, 1.0e4))
    sol = solve(prob, Rosenbrock23())
    @test sol.u[1] isa ComponentArray
end

@testset "Issue 53" begin
    x0 = ComponentArray(x = ones(10))
    prob = ODEProblem((u, p, t) -> u, x0, (0.0, 1.0))
    # Sundials CVODE_BDF doesn't support ComponentArrays directly (NVector conversion fails)
    # Tracking: https://github.com/SciML/ComponentArrays.jl/issues/332
    @test_broken begin
        sol =
            solve(prob, CVODE_BDF(linear_solver = :BCG), reltol = 1.0e-15, abstol = 1.0e-15)
        sol(1)[1] ≈ exp(1)
    end
end

@testset "Issue 55" begin
    f!(D, x, p, t) = nothing
    x0 = ComponentArray(x = zeros(4))
    prob = ODEProblem(f!, x0, (0.0, 1.0), 0.0)
    sol = solve(prob, Rodas4())
    @test sol.u[1] == x0
end

# @testset "Unitful" begin
#     tspan = (0.0u"s", 10.0u"s")
#     pos = 0.0u"m"
#     vel = 0.0u"m/s"
#     x0 = ComponentArray{Union{typeof(pos), typeof(vel)}}(; pos, vel)
#     F(t) = 1

#     # double integrator in state-space form
#     A = Union{typeof(0u"s^-1"), typeof(0u"s^-2"), Int}[0u"s^-1" 1; 0u"s^-2" 0u"s^-1"]
#     B = Union{typeof(0u"m/s"), typeof(1u"m/s^2")}[0u"m/s"; 1u"m/s^2"]
#     di(x,u,t) = A*x .+ B*u(t)

#     prob = ODEProblem(di, x0, tspan, F)
#     sol = solve(prob, Tsit5())
#     @test unit(sol[end].pos) == u"m"
#     @test unit(sol[end].vel) == u"m/s"
# end

# `Pkg.test(coverage = true)` instruments only this package's source, which skews
# the ComponentArray vs Array timing ratios, so measure them in an uninstrumented process.
@testset "Performance" begin
    julia = filter(!startswith("--code-coverage"), Base.julia_cmd().exec)
    cmd = `$julia --project=$(Base.active_project()) $(joinpath(@__DIR__, "perf", "performance.jl"))`
    cmd = addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ';' : ':'))
    @test success(pipeline(cmd; stdout, stderr))
end

@testset "SymbolicIndexingInterface solution indexing (SciML/DifferentialEquations.jl#957)" begin
    using SymbolicIndexingInterface:
        is_variable, variable_index, variable_symbols, is_parameter, parameter_symbols

    function lorenz!(du, u, p, t)
        du.x = p.a * (u.y - u.x)
        du.y = u.x * (p.b - u.z) - u.y
        du.z = u.x * u.y - p.c * u.z
        return nothing
    end

    u0 = ComponentVector(x = 1.0, y = 0.0, z = 0.0)
    p = ComponentVector(a = 10.0, b = 28.0, c = 8 / 3)
    prob = ODEProblem(lorenz!, u0, (0.0, 1.0), p)
    sol = solve(prob, Tsit5())

    @test is_variable(sol, :x)
    @test variable_index(sol, :x) === :x
    @test variable_symbols(sol) == [:x, :y, :z]
    @test sol[:x][1] == 1.0
    @test sol(0.1; idxs = :x) isa Real
    @test sol(0.0:0.1:0.3; idxs = :x) isa AbstractVector
    @test size(Array(sol(0.0:0.1:0.3; idxs = [:x, :y]))) == (2, 4)
    @test is_parameter(sol, :a)
    @test parameter_symbols(sol) == [:a, :b, :c]
    @test sol.ps[:a] == 10.0
    @test !is_variable(sol, :a)
    @test !is_parameter(sol, :x)

    # Nested top-level components remain indexable by name
    function nest!(du, u, p, t)
        du.a = -u.a
        du.b.x = -u.b.x
        du.b.y = -u.b.y
        return nothing
    end
    un = ComponentVector(a = 1.0, b = (x = 2.0, y = 3.0))
    soln = solve(ODEProblem(nest!, un, (0.0, 1.0)), Tsit5())
    @test is_variable(soln, :b)
    @test soln[:a][1] == 1.0
    @test soln[:b][1].x == 2.0

    # Explicit `sys` on the ODEFunction is not overridden
    using SymbolicIndexingInterface: SymbolCache
    sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t)
    f = ODEFunction(lorenz!; sys = sys)
    sol_sys = solve(ODEProblem(f, u0, (0.0, 0.1), p), Tsit5())
    @test sol_sys[:x][1] == 1.0
    @test variable_index(sol_sys, :x) == 1
end
