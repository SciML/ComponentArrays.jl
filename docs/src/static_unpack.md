```@meta
DocTestSetup = quote
    using ComponentArrays, StaticArrays
end
```
# Unpacking to StaticArrays

Often `ComponentArray`s will hold vector or matrix components for which the user will want to unpack and operate on. If these arrays are small (for example, 3-vectors of positions and velocities), it is usually useful to convert them to `SArray`s from [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) before doing operations on them to avoid heap allocations. For this reason, we export a `@static_unpack` macro that works similarly to `@unpack` from [`UnPack.jl`](https://github.com/mauro3/UnPack.jl), except it converts all plain arrays to `SArray`s. Anything that isn't a plain array (for example, scalars or inner `ComponentArray`s) will be unpacked as usual.

## Example
```jldoctest
julia> x = ComponentVector(a=5, b=[4, 1], c = [1 2; 3 4], d=(e=2, f=[6, 30.0]));

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

## Why not just have static ComponentArrays?
Most of the places `ComponentVector`s are used, the top-level `ComponentVector` is too large to be efficiently backed by a `SVector`. Typically you only need the inner components to be `SVector`s for fast operations (for example, simulating rigid body dynamics where states and derivatives are mostly 3-vectors that need to undergo rotations and translations).

Also, just from experience, immutable `ComponentArray`s tend to be really clunky to work with.
