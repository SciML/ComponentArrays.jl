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

end
