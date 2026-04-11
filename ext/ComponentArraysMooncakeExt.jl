module ComponentArraysMooncakeExt

using ComponentArrays, Mooncake

# ComponentVector handling in @from_rrule
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{@NamedTuple{data::A, axes::Mooncake.NoFData}},
        r::Mooncake.NoRData,
        t::A,
    ) where {P <: Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}, A <: Array{P}}
    return Mooncake.increment_and_get_rdata!(f.data[:data], r, t)
end

# Same path, but the upstream cotangent has already been wrapped back into a
# `ComponentVector` (e.g. by a `@from_chainrules`/`@from_rrule` rule whose
# rrule returned the gradient as a `ComponentVector` instead of the raw
# backing `Vector`).  Strip the wrapper and accumulate into the underlying
# data buffer that the FData is tracking.
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{@NamedTuple{data::A, axes::Mooncake.NoFData}},
        r::Mooncake.NoRData,
        t::ComponentVector{P, A},
    ) where {P <: Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}, A <: Array{P}}
    return Mooncake.increment_and_get_rdata!(f.data[:data], r, getdata(t))
end

function Mooncake.friendly_tangent_cache(x::ComponentArray)
    Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}(copy(x))
end

end
