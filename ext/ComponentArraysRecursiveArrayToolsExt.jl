module ComponentArraysRecursiveArrayToolsExt

using ComponentArrays
using RecursiveArrayTools: RecursiveArrayTools

AVOA = RecursiveArrayTools.AbstractVectorOfArray

function Base.Array(va::AVOA{T, N, A}) where {T, N, A <: AbstractVector{<:ComponentVector}}
    return ComponentArray(reduce(hcat, va.u), only(getaxes(va.u[1])), FlatAxis())
end

end
