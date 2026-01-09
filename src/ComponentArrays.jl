module ComponentArrays

import ChainRulesCore
import StaticArrayInterface, ArrayInterface, Functors
import ConstructionBase
import Adapt

using LinearAlgebra
using StaticArraysCore: StaticArray, SArray, SVector, SMatrix

const FlatIdx = Union{Integer, CartesianIndex, CartesianIndices, AbstractArray{<:Integer}}
const FlatOrColonIdx = Union{FlatIdx, Colon}

include("utils.jl")
export fastindices # Deprecated

include("lazyarray.jl")

include("axis.jl")
export AbstractAxis, Axis, PartitionedAxis, ShapedAxis, Shaped1DAxis, ViewAxis, FlatAxis

include("componentarray.jl")
export ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata, valkeys
export update_component_array

include("componentindex.jl")
export KeepIndex

include("array_interface.jl")
# Base methods: parent, size, elsize, axes, reinterpret, hcat, vcat, permutedims, IndexStyle, to_indices, to_index, getindex, setindex!, view, pointer, unsafe_convert, strides, stride
# StaticArrayInterface methods: strides, size, parent_type

include("linear_algebra.jl")
# Base methods: *, \, /
# StaticArrayInterface methods: lu_instance

include("namedtuple_interface.jl")
# Base methods: hash, ==, keys, haskey, propertynames, getproperty, setproperty!

include("similar_convert_copy.jl")
# Base methods: similar, zero, copy, copyto!, deepcopy, convert (to Array and NamedTuple), promote

include("broadcasting.jl")
# Base methods: BroadcastStyle, convert(to Broadcasted{Nothing}), similar, map, dataids
# Broadcast methods: BroadcastStyle, broadcasted, broadcast_unalias

include("show.jl")

include("plot_utils.jl")
export labels, label2index

include("compat/chainrulescore.jl")

include("compat/static_arrays.jl")
export @static_unpack

include("compat/functors.jl")

end
