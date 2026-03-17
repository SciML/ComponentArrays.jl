

module ComponentArraysMooncakeExt

using ComponentArrays, Mooncake

# ComponentVector handling in @from_rrule
function increment_and_get_rdata!(
    f::Mooncake.FData{@NamedTuple{data::Vector{T},axes::Mooncake.NoFData}},
    r::Mooncake.NoRData,
    t::Vector{T},
) where {T<:Base.IEEEFloat}
    f.data[:data] .+= t
    return r
end

end
