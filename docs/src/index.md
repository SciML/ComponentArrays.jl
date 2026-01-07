# ComponentArrays.jl

The main export of this package is the `ComponentArray` type. "Components" of `ComponentArray`s
are really just array blocks that can be accessed through a named index. The magic here is
that this named indexing can create a new `ComponentArray` whose data is a view into the original,
allowing for standalone models to be composed together by simple function composition. In
essence, `ComponentArray`s allow you to do the things you would usually need a modeling
language for, but without actually needing a modeling language. The main targets are for use
in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) and
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), but anything that requires
flat vectors is fair game.
