using ComponentArrays
using DifferentialEquations
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
    @test sol[1] isa ComponentArray
end

@testset "Issue 53" begin
    x0 = ComponentArray(x = ones(10))
    prob = ODEProblem((u, p, t) -> u, x0, (0.0, 1.0))
    sol = solve(prob, CVODE_BDF(linear_solver = :BCG), reltol = 1.0e-15, abstol = 1.0e-15)
    @test sol(1)[1] ≈ exp(1)
end

@testset "Issue 55" begin
    f!(D, x, p, t) = nothing
    x0 = ComponentArray(x = zeros(4))
    prob = ODEProblem(f!, x0, (0.0, 1.0), 0.0)
    sol = solve(prob, Rodas4())
    @test sol[1] == x0
end

# @testset "Unitful" begin
#     tspan = (0.0u"s", 10.0u"s")
#     pos = 0.0u"m"
#     vel = 0.0u"m/s"
#     x0 = ComponentArray{Union{typeof(pos), typeof(vel)}}(pos=pos, vel=vel)
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
        solve(lprob1, Rodas5(autodiff = false))
        solve(cprob1, Rodas5())
        solve(cprob1, Rodas5(autodiff = false))

        ltime1 = @elapsed lsol1 = solve(lprob1, Rodas5())
        ltime2 = @elapsed lsol2 = solve(lprob1, Rodas5(autodiff = false))
        ctime1 = @elapsed csol1 = solve(cprob1, Rodas5())
        ctime2 = @elapsed csol2 = solve(cprob1, Rodas5(autodiff = false))

        @test (ctime1 - ltime1) / ltime1 < 0.05
        @test (ctime2 - ltime2) / ltime2 < 0.05
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

        @test (ctime - time) / time < 0.1
        @test (ctime - ltime) / ltime < 0.05
    end
end
