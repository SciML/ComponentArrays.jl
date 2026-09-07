using Reactant, ComponentArrays
using Test

x = ComponentArray(; a = rand(4), b = rand(2))
x_ra = Reactant.to_rarray(x)

fn(x) = x.a .+ sum(abs2, x.b) .+ 1

@test @jit(fn(x_ra)) ≈ fn(x)
