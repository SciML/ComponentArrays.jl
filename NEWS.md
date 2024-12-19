# ComponentArrays.jl NEWS
Notes on new features (minor releases). For more details on bugfixes and non-feature-adding changes (patch releases), check out the [releases page](https://github.com/SciML/ComponentArrays.jl/releases).

### v0.15.0
- Unpack array components as `StaticArray`s!
```julia
julia> x = ComponentArray(a=5, b=[4, 1], c = [1 2; 3 4], d=(e=2, f=[6, 30.0]));

julia> @static_unpack a, b, c, d = x;

julia> a
5.0

julia> b
2-element SVector{2, Float64} with indices SOneTo(2):
 4.0
 1.0

julia> c
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 1.0  2.0
 3.0  4.0

julia> d
ComponentVector{Float64,SubArray...}(e = 2.0, f = [6.0, 30.0])
```

### v0.12.0
- Multiple symbol indexing!
  - Use either an `Array` or `Tuple` of `Symbol`s to extract multiple named components into a new `ComponentArray`
  - It's fast!
```julia
julia> ca = ComponentArray(a=5, b=[4, 1], c=(a=2, b=[6, 30.0]))
ComponentVector{Float64}(a = 5.0, b = [4.0, 1.0], c = (a = 2.0, b = [6.0, 30.0]))

julia> ca[(:c, :a)]
ComponentVector{Float64}(c = (a = 2.0, b = [6.0, 30.0]), a = 5.0)

julia> ca[[:c, :a]] == ca[(:c, :a)]
true

julia> @view ca[(:c, :a)]
ComponentVector{Float64,SubArray...}(c = (a = 2.0, b = [6.0, 30.0]), a = 5.0)
```
### v0.11.0
- Calling `axes` on a `ComponentArray` returns a new `CombinedAxis` type!
  - Doing things The Right Way™!
  - No more complicated and error-prone custom broadcasting machinery!
  - No more weird special cases!
### v0.10.0
- All indexing now slices rather than sometimes viewing and sometimes slicing!
- Property access methods (i.e. "dot-access") still use views!
```julia
julia> x = ComponentArray(a=1, b=[4,2])
ComponentVector{Int64}(a = 1, b = [4, 2])

julia> x.b # Dot-access still views by default
2-element view(::Vector{Int64}, 2:3) with eltype Int64:
 4
 2

julia> x[:b] # Slicing now slices
2-element Vector{Int64}:
 4
 2

julia> @view x[:b] # Use @view to view
2-element view(::Vector{Int64}, 2:3) with eltype Int64:
 4
 2
```

### v0.9.0
- Construct `ComponentArray`s from `Dict`s!
```julia
julia> d = Dict(:a=>rand(3), :b=>rand(2,2))
Dict{Symbol, Array{Float64, N} where N} with 2 entries:
  :a => [0.996693, 0.148683, 0.203083]
  :b => [0.68759 0.41585; 0.900591 0.377475]

julia> ComponentArray(d)
ComponentVector{Float64}(a = [0.9966932920820444, 0.14868304847436709, 0.20308284992079573], b = [0.6875902095731583 0.415850281435181; 0.9005909643364229 0.3774747843717925])
```

### v0.8.0
- Generated `valkeys` function for fast iteration over `ComponentVector` subcomponents!
```julia
  julia> ca = ComponentArray(a=1, b=[1,2,3], c=(a=4,))
  ComponentVector{Int64}(a = 1, b = [1, 2, 3], c = (a = 4))
  
  julia> valkeys(ca)
  (Val{:a}(), Val{:b}(), Val{:c}())

  julia> [ca[k] for k in valkeys(ca)]
  3-element Array{Any,1}:
   1
    [1, 2, 3]
    ComponentVector{Int64,SubArray...}(a = 4)
  
  julia> @btime sum(prod($ca[k]) for k in valkeys($ca))
    11.511 ns (0 allocations: 0 bytes)
  11
```

### v0.7.0
- Much faster (and lazier) arrays of subcomponents!
```julia
julia> ca = ComponentArray(a=5, b=(a=zeros(4,4), b=0), c=(a=[(a=1, b=2), (a=3, b=1), (a=1, b=2), (a=3, b=1)], b=[1., 2., 4]));

julia> @btime sum(x.a + x.b for x in $ca.c.a);
  127.160 ns (2 allocations: 480 bytes)

julia> @btime sum(x.a + x.b for x in $ca.c.a);
  36.895 ns (0 allocations: 0 bytes)
```

### v0.6.0
- Easier DifferentialEquations plotting!
    - Automatic legend labeling!
    - `Symbol` and `String` support for the `vars` plot keyword!
    - See it in an action [here](https://github.com/SciML/ComponentArrays.jl/blob/main/docs/src/examples/adaptive_control.md)!

### v0.5.0
- Constructor for making new `ComponentVector`s with additional fields! Watch out, it's slow!
```julia
julia> x = ComponentArray(a=5, b=[1, 2])
ComponentVector{Int64}(a = 5, b = [1, 2])

julia> moar_x = ComponentArray(x; c=zeros(2,2), d=(a=2, b=10))
ComponentVector{Int64}(a = 5, b = [1, 2], c = [0 0; 0 0], d = (a = 2, b = 10))
```

### v0.4.0
- Zygote rules for DiffEqFlux support! Check out [the docs](https://sciml.github.io/ComponentArrays.jl/dev/examples/DiffEqFlux/) for an example!

### v0.3.0
- Matrix and higher-dimensional array components!

...and plenty more!
