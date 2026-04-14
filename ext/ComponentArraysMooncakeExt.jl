module ComponentArraysMooncakeExt

using ComponentArrays, Mooncake
using Base: IEEEFloat

const _FloatLike = Union{IEEEFloat, Complex{<:IEEEFloat}}

# === Flat-Array-backed ComponentVector fdata ==========================================
# `Mooncake.FData{@NamedTuple{data::A, axes::NoFData}}` is the fdata layout of a
# `ComponentArray{T, N, A<:Array, Axes}` — the common "owns its storage" case.
#
# We need to handle three incoming ChainRules cotangent shapes that arise from
# `@from_rrule` / `@from_chainrules` declarations:
#   (a) a raw `Array{P}` matching the primal underlying storage,
#   (b) a `ComponentArray` with the same underlying storage type,
#   (c) a `ComponentArray` whose data field is a different `AbstractArray{P}`
#       (e.g. a `SubArray` produced by projecting a parent cotangent).

# (a) raw Array cotangent
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{@NamedTuple{data::A, axes::Mooncake.NoFData}},
        r::Mooncake.NoRData,
        t::A,
    ) where {P <: _FloatLike, A <: Array{P}}
    return Mooncake.increment_and_get_rdata!(f.data[:data], r, t)
end

# (b) / (c) ComponentArray cotangent against a flat-Array-backed primal
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{@NamedTuple{data::A, axes::Mooncake.NoFData}},
        r::Mooncake.NoRData,
        t::ComponentArray{P, N, <:AbstractArray{P}},
    ) where {P <: _FloatLike, N, A <: Array{P}}
    data_t = getdata(t)
    t_vec = data_t isa Array{P} ? data_t : collect(data_t)
    return Mooncake.increment_and_get_rdata!(f.data[:data], r, t_vec)
end

# === SubArray-backed ComponentVector fdata ============================================
# A `ComponentVector` produced by `getproperty(::ComponentVector, ::Symbol)` (and any
# other view-producing path) wraps a `SubArray` rather than a `Vector`. Its Mooncake
# fdata accordingly nests an inner `FData` describing the SubArray's fields.
#
# We can only aggregate a ChainRules cotangent into this layout when the view fully
# covers its parent — otherwise the unmodelled indices leave us unable to place the
# cotangent into the correct slice of the parent tangent. That "full cover" case is
# however the common one: sub-CVs that land at an `@from_rrule` boundary are usually
# freshly allocated and own all of their parent storage. Outside of that, we raise a
# clear error instead of silently corrupting gradients.

function _increment_subarray_fdata!(f_cv, t_data::AbstractArray{P}) where {P <: _FloatLike}
    parent = f_cv.data[:data].data[:parent]
    if length(t_data) != length(parent)
        throw(
            ArgumentError(
                "ComponentArraysMooncakeExt: cannot aggregate a cotangent of length " *
                    "$(length(t_data)) into a SubArray-backed ComponentVector tangent whose " *
                    "parent has length $(length(parent)). This happens when a cotangent " *
                    "flows into a view that does not fully cover its parent; there is no " *
                    "way to recover the view indices from Mooncake fdata alone. Please " *
                    "file an issue against ComponentArrays.jl with a reproducer so the " *
                    "offending rrule can be patched.",
            ),
        )
    end
    t_vec = t_data isa Array{P} ? t_data : collect(t_data)
    Mooncake.increment_and_get_rdata!(parent, Mooncake.NoRData(), t_vec)
    return Mooncake.NoRData()
end

function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{
            @NamedTuple{
                data::Mooncake.FData{
                    @NamedTuple{
                        parent::Array{P, 1},
                        indices::Mooncake.NoFData,
                        offset1::Mooncake.NoFData,
                        stride1::Mooncake.NoFData,
                    },
                },
                axes::Mooncake.NoFData,
            },
        },
        r::Mooncake.NoRData,
        t::Array{P},
    ) where {P <: _FloatLike}
    return _increment_subarray_fdata!(f, t)
end

function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{
            @NamedTuple{
                data::Mooncake.FData{
                    @NamedTuple{
                        parent::Array{P, 1},
                        indices::Mooncake.NoFData,
                        offset1::Mooncake.NoFData,
                        stride1::Mooncake.NoFData,
                    },
                },
                axes::Mooncake.NoFData,
            },
        },
        r::Mooncake.NoRData,
        t::ComponentArray{P, N, <:AbstractArray{P}},
    ) where {P <: _FloatLike, N}
    return _increment_subarray_fdata!(f, getdata(t))
end

function Mooncake.friendly_tangent_cache(x::ComponentArray)
    return Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}(copy(x))
end

end
