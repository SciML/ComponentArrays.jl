# Interfaces

## Abstract axes

`ComponentArrays` uses [`AbstractAxis`](@ref) values to associate named components with
flat positions in an array. A custom axis subtype is appropriate when the component map
is static and encoded in the subtype. Ordinary named layouts should use [`Axis`](@ref).

An `AbstractAxis` subtype must provide an `IdxMap` type parameter whose keys are the
accepted component names and whose values describe the corresponding flat positions or
nested component metadata. The map is part of the type, so instances should not carry a
second runtime copy of it.

The generic implementation supplies `keys`, symbol and `Val` indexing, multi-name
indexing, `firstindex`, `lastindex`, and `valkeys`. A custom subtype should therefore
usually only define the subtype and a zero-argument constructor. Its map must describe
the same layout as the data passed to [`ComponentArray`](@ref).

```@example abstract_axis
using ComponentArrays

struct TwoComponentAxis <: AbstractAxis{(left = 1, right = 2)} end

axis = TwoComponentAxis()
keys(axis), axis[Val(:left)], valkeys(axis)
```

The package tests this contract with an external subtype that defines no indexing or
iteration methods of its own. This is the intended extension pattern for the interface.
