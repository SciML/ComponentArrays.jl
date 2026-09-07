using ComponentArrays
using DifferentialEquations
using UnPack: @unpack

tspan = (0.0, 20.0)

## Lorenz system
function lorenz!(D, u, p, t; f = 0.0)
    @unpack σ, ρ, β = p
    @unpack x, y, z = u

    D.x = σ * (y - x)
    D.y = x * (ρ - z) - y - f
    D.z = x * y - β * z
    return nothing
end
function lorenz_jac!(D, u, p, t)
    @unpack σ, ρ, β = p
    @unpack x, y, z = u

    D[:x, :x] = -σ
    D[:x, :y] = σ

    D[:y, :x] = ρ
    D[:y, :y] = -1
    D[:y, :z] = -x

    D[:z, :x] = y
    D[:z, :y] = x
    D[:z, :z] = -β
    return nothing
end

lorenz_p = (σ = 10.0, ρ = 28.0, β = 8 / 3)
lorenz_ic = ComponentArray(x = 0.0, y = 0.0, z = 0.0)
lorenz_fun = ODEFunction(lorenz!, jac = lorenz_jac!)
lorenz_prob = ODEProblem(lorenz_fun, lorenz_ic, tspan, lorenz_p)

## Lotka-Volterra system
function lotka!(D, u, p, t; f = 0.0)
    @unpack α, β, γ, δ = p
    @unpack x, y = u

    D.x = α * x - β * x * y + f
    D.y = -γ * y + δ * x * y
    return nothing
end
function lotka_jac!(D, u, p, t)
    @unpack α, β, γ, δ = p
    @unpack x, y = u

    D[:x, :x] = α - β * y
    D[:x, :y] = -β * x

    D[:y, :x] = δ * y
    D[:y, :y] = -γ + δ * x
    return nothing
end

lotka_p = (α = 2 / 3, β = 4 / 3, γ = 1.0, δ = 1.0)
lotka_ic = ComponentArray(x = 1.0, y = 1.0)
lotka_fun = ODEFunction(lotka!, jac = lotka_jac!)
lotka_prob = ODEProblem(lotka_fun, lotka_ic, tspan, lotka_p)

## Composed Lorenz and Lotka-Volterra system
function composed!(D, u, p, t)
    c = p.c #coupling parameter
    @unpack lorenz, lotka = u

    lorenz!(D.lorenz, lorenz, p.lorenz, t, f = c * lotka.x)
    lotka!(D.lotka, lotka, p.lotka, t, f = c * lorenz.x)
    return nothing
end
function composed_jac!(D, u, p, t)
    c = p.c
    @unpack lorenz, lotka = u

    lorenz_jac!(@view(D[:lorenz, :lorenz]), lorenz, p.lorenz, t)
    lotka_jac!(@view(D[:lotka, :lotka]), lotka, p.lotka, t)

    @view(D[:lorenz, :lotka])[:y, :x] = -c
    @view(D[:lotka, :lorenz])[:x, :x] = c
    return nothing
end

comp_p = (lorenz = lorenz_p, lotka = lotka_p, c = 0.01)
comp_ic = ComponentArray(lorenz = lorenz_ic, lotka = lotka_ic)
comp_fun = ODEFunction(composed!, jac = composed_jac!)
comp_prob = ODEProblem(comp_fun, comp_ic, tspan, comp_p)

## Solve problem
# We can solve the composed system...
comp_sol = solve(comp_prob, Rodas5())

# ...or we can unit test one of the component systems
lotka_sol = solve(lotka_prob, Rodas5())
