using ComponentArrays, BenchmarkTools
using LinearAlgebra, StableRNGs
using ADTypes: AutoFiniteDiff
using LabelledArrays
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqTsit5

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# =============================================================================
# ComponentVector
# =============================================================================

SUITE["componentvector"] = BenchmarkGroup()

a_data = rand(rng, 100)
b_c = rand(rng, 50)
b_d = rand(rng, 10, 10)
cv = ComponentArray(a = a_data, b = (c = b_c, d = b_d))
cv2 = ComponentArray(a = rand(rng, 100), b = (c = rand(rng, 50), d = rand(rng, 10, 10)))

SUITE["componentvector"]["construct"] = @benchmarkable ComponentArray(
    a = $a_data, b = (c = $b_c, d = $b_d)
)
SUITE["componentvector"]["getproperty"] = @benchmarkable $cv.b.c
SUITE["componentvector"]["getindex_symbol"] = @benchmarkable $cv[Val(:b)]
SUITE["componentvector"]["broadcast_add"] = @benchmarkable $cv .+ $cv2
SUITE["componentvector"]["norm"] = @benchmarkable norm($cv)
SUITE["componentvector"]["map"] = @benchmarkable map(sin, $cv)

# =============================================================================
# ComponentMatrix
# =============================================================================

SUITE["componentmatrix"] = BenchmarkGroup()

cm_data = rand(rng, 20, 20)
cm_ax = Axis(x = 1:10, y = 11:20)
cm_ax2 = Axis(u = 1:10, v = 11:20)
cm = ComponentMatrix(cm_data, cm_ax, cm_ax2)
cv_20 = ComponentVector(x = rand(rng, 10), y = rand(rng, 10))

SUITE["componentmatrix"]["construct"] = @benchmarkable ComponentMatrix(
    $cm_data, $cm_ax, $cm_ax2
)
SUITE["componentmatrix"]["getindex_symbol"] = @benchmarkable $cm[Val(:x), Val(:u)]
SUITE["componentmatrix"]["matmul"] = @benchmarkable $cm * $cv_20

# =============================================================================
# Views
# =============================================================================

SUITE["views"] = BenchmarkGroup()
SUITE["views"]["view_axis"] = @benchmarkable $cv[Val(:b)]
SUITE["views"]["getdata"] = @benchmarkable getdata($cv)

# =============================================================================
# ODE solves
# =============================================================================

SUITE["ode"] = BenchmarkGroup()

function issue36_f1(du, u, p, t)
    du.x .= -1 .* u.x .* u.y .* p[1]
    return du.y .= -1 .* u.y .* p[2]
end

let
    n = 1000
    p = [0.1, 0.1]
    lu_0 = @LArray fill(1000.0, 2 * n) (x = (1:n), y = ((n + 1):(2 * n)))
    cu_0 = ComponentArray(x = fill(1000.0, n), y = fill(1000.0, n))
    issue36_lprob = ODEProblem(issue36_f1, lu_0, (0, 100.0), p)
    issue36_cprob = ODEProblem(issue36_f1, cu_0, (0, 100.0), p)
    for (name, alg) in (
            "Rodas5" => Rodas5(),
            "Rodas5_finitediff" => Rodas5(autodiff = AutoFiniteDiff()),
        )
        SUITE["ode"]["issue_36"][name] = BenchmarkGroup()
        SUITE["ode"]["issue_36"][name]["ComponentArray"] = @benchmarkable solve(
            $issue36_cprob, $alg
        )
        SUITE["ode"]["issue_36"][name]["LArray"] = @benchmarkable solve(
            $issue36_lprob, $alg
        )
    end
end

const HEAT_NKNOTS = 100
const HEAT_H² = (1.0 / (HEAT_NKNOTS + 1))^2
function heat_conduction(du, u, p, t)
    u₃ = @view u[3:end]
    u₂ = @view u[2:(end - 1)]
    u₁ = @view u[1:(end - 2)]
    @. du[2:(end - 1)] = (u₃ - 2 * u₂ + u₁) / HEAT_H²
    return nothing
end

let
    u0 = randn(rng, 300)
    u0_ca = ComponentArray(a = u0[1:100], b = u0[101:200], c = u0[201:300])
    u0_la = @LArray u0 (a = 1:100, b = 101:200, c = 201:300)
    heat_cprob = ODEProblem(heat_conduction, u0_ca, (0.0, 1.0))
    heat_lprob = ODEProblem(heat_conduction, u0_la, (0.0, 1.0))
    heat_prob = ODEProblem(heat_conduction, u0, (0.0, 1.0))
    SUITE["ode"]["heat_conduction"]["ComponentArray"] = @benchmarkable solve(
        $heat_cprob, Tsit5(), saveat = 0.2
    )
    SUITE["ode"]["heat_conduction"]["LArray"] = @benchmarkable solve(
        $heat_lprob, Tsit5(), saveat = 0.2
    )
    SUITE["ode"]["heat_conduction"]["Array"] = @benchmarkable solve(
        $heat_prob, Tsit5(), saveat = 0.2
    )
end
