using ComponentArrays
using ADTypes: AutoFiniteDiff
using DifferentialEquations
using OrdinaryDiffEqRosenbrock
using LabelledArrays
using Test

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

        lu_0 = @LArray fill(1000.0, 2 * n) (x = (1:n), y = ((n + 1):(2 * n)))
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
            u₂ = @view u[2:(end - 1)]
            u₁ = @view u[1:(end - 2)]
            @. du[2:(end - 1)] = (u₃ - 2 * u₂ + u₁) / h²
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
