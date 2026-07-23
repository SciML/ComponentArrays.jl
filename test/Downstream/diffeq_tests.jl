using ComponentArrays
using ADTypes: AutoFiniteDiff
using DifferentialEquations
using OrdinaryDiffEqRosenbrock
using LabelledArrays
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

# Performance tests use relaxed thresholds for CI (shared runners have noisy timing).
# These tests catch catastrophic regressions, not subtle overhead.
@testset "Performance" begin
    @testset "Issue 36" begin
        function f1(du, u, p, t)
            du.x .= -1 .* u.x .* u.y .* p[1]
            du.y .= -1 .* u.y .* p[2]
        end

        n = 1000

        p = [0.1, 0.1]

        lu_0 = @LArray fill(1000.0, 2 * n) (x = (1:n), y = ((n+1):(2*n)))
        cu_0 = ComponentArray(x = fill(1000.0, n), y = fill(1000.0, n))

        lprob1 = ODEProblem(f1, lu_0, (0, 100.0), p)
        cprob1 = ODEProblem(f1, cu_0, (0, 100.0), p)

        solve(lprob1, Rodas5())
        solve(lprob1, Rodas5(autodiff = AutoFiniteDiff()))
        solve(cprob1, Rodas5())
        solve(cprob1, Rodas5(autodiff = AutoFiniteDiff()))

        ltime1 = @elapsed lsol1 = solve(lprob1, Rodas5())
        ltime2 = @elapsed lsol2 = solve(lprob1, Rodas5(autodiff = AutoFiniteDiff()))
        ctime1 = @elapsed csol1 = solve(cprob1, Rodas5())
        ctime2 = @elapsed csol2 = solve(cprob1, Rodas5(autodiff = AutoFiniteDiff()))

        # Issue 36 perf check: ComponentVector solve overhead vs plain Vector.
        # Threshold is generous because self-hosted runner timing varies wildly:
        # consecutive reruns on different machines observed 10.3x, 12.4x, and
        # 15.5x — i.e. up to ~50% spread from the same code. A tight threshold
        # here catches noise, not regressions. This assertion guards against
        # pathological blow-ups (>20x); finer perf tracking belongs in a
        # dedicated benchmark suite. See SciML/ComponentArrays.jl#36.
        @test (ctime1 - ltime1) / ltime1 < 20.0
        @test (ctime2 - ltime2) / ltime2 < 20.0
    end

    @testset "Slack Issue 2021-2-19" begin
        nknots = 100
        h² = (1.0 / (nknots + 1))^2
        function heat_conduction(du, u, p, t)
            u₃ = @view u[3:end]
            u₂ = @view u[2:(end-1)]
            u₁ = @view u[1:(end-2)]
            @. du[2:(end-1)] = (u₃ - 2 * u₂ + u₁) / h²
            nothing
        end

        t0, t1 = 0.0, 1.0
        u0 = randn(300)
        u0_ca = ComponentArray(a = u0[1:100], b = u0[101:200], c = u0[201:300])
        u0_la = @LArray u0 (a = 1:100, b = 101:200, c = 201:300)

        cprob = ODEProblem(heat_conduction, u0_ca, (t0, t1))
        lprob = ODEProblem(heat_conduction, u0_la, (t0, t1))
        prob = ODEProblem(heat_conduction, u0, (t0, t1))

        solve(cprob, Tsit5(), saveat = 0.2)
        solve(lprob, Tsit5(), saveat = 0.2)
        solve(prob, Tsit5(), saveat = 0.2)

        ctime = @elapsed solve(cprob, Tsit5(), saveat = 0.2)
        ltime = @elapsed solve(lprob, Tsit5(), saveat = 0.2)
        time = @elapsed solve(prob, Tsit5(), saveat = 0.2)

        @test (ctime - time) / time < 10.0
        @test (ctime - ltime) / ltime < 10.0
    end
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
