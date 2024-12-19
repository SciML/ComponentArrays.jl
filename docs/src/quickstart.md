```@meta
DocTestSetup = quote
    using ComponentArrays
end
```
# Quick Start

## General use
The easiest way to construct 1-dimensional `ComponentArray`s is as if they were `NamedTuple`s. In fact, a good way to think about them is as arbitrarily nested, mutable `NamedTuple`s that can be passed through a solver.
```jldoctest quickstart
julia> c = (a=2, b=[1, 2]);

julia> x = ComponentArray(a=1.0, b=[2, 1, 4], c=c)
ComponentVector{Float64}(a = 1.0, b = [2.0, 1.0, 4.0], c = (a = 2.0, b = [1.0, 2.0]))

julia> x.c.a = 400; x
ComponentVector{Float64}(a = 1.0, b = [2.0, 1.0, 4.0], c = (a = 400.0, b = [1.0, 2.0]))

julia> x[5]
400.0

julia> collect(x)
7-element Vector{Float64}:
   1.0
   2.0
   1.0
   4.0
 400.0
   1.0
   2.0

julia> typeof(similar(x, Int32)) === typeof(ComponentVector{Int32}(a=1, b=[2, 1, 4], c=c))
true
```
`ComponentArray`s can be constructed from existing
`ComponentArray`s (currently nested fields cannot be changed this way):
```jldoctest
julia> v = ComponentVector(a=1, b=2, c=3);

julia> ComponentVector(v; a=11, new=42)
ComponentVector{Int64}(a = 11, b = 2, c = 3, new = 42)
```

Higher dimensional `ComponentArray`s can be created too, but it's a little messy at the moment. The nice thing for modeling is that dimension expansion through broadcasted operations can create higher-dimensional `ComponentArray`s automatically, so Jacobian cache arrays that are created internally with `false .* x .* x'` will be `ComponentArray`s with proper axes. Check out the [ODE with Jacobian](https://github.com/SciML/ComponentArrays.jl/blob/master/examples/ODE_jac_example.jl) example in the examples folder to see how this looks in practice.
```jldoctest quickstart
julia> x2 = x .* x'
7×7 ComponentMatrix{Float64} with axes Axis(a = 1, b = 2:4, c = ViewAxis(5:7, Axis(a = 1, b = 2:3))) × Axis(a = 1, b = 2:4, c = ViewAxis(5:7, Axis(a = 1, b = 2:3)))
   1.0    2.0    1.0     4.0     400.0    1.0    2.0
   2.0    4.0    2.0     8.0     800.0    2.0    4.0
   1.0    2.0    1.0     4.0     400.0    1.0    2.0
   4.0    8.0    4.0    16.0    1600.0    4.0    8.0
 400.0  800.0  400.0  1600.0  160000.0  400.0  800.0
   1.0    2.0    1.0     4.0     400.0    1.0    2.0
   2.0    4.0    2.0     8.0     800.0    2.0    4.0

julia> x2[:c,:c]
3×3 ComponentMatrix{Float64} with axes Axis(a = 1, b = 2:3) × Axis(a = 1, b = 2:3)
 160000.0  400.0  800.0
    400.0    1.0    2.0
    800.0    2.0    4.0

julia> x2[:a,:a]
1.0

julia> x2[:a,:c]
ComponentVector{Float64}(a = 400.0, b = [1.0, 2.0])

julia> x2[:b,:c]
3×3 ComponentMatrix{Float64} with axes FlatAxis() × Axis(a = 1, b = 2:3)
  800.0  2.0  4.0
  400.0  1.0  2.0
 1600.0  4.0  8.0
```
