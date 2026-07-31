module ComponentArraysMooncakeExt

using ComponentArrays
using Mooncake: Mooncake
using Base: IEEEFloat
using Random: AbstractRNG, randn!
using LinearAlgebra: dot

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

# === Tangent → ComponentArray gradient copy ===========================================
# `DifferentiationInterface.value_and_gradient!(::AutoMooncake, …)` writes the gradient
# into a user-supplied `ComponentArray` buffer with an unconditional
# `copyto!(grad, new_grad)`. Mooncake's `tangent_type` for a `ComponentArray` is a
# `Mooncake.Tangent` struct, which is not an `AbstractArray` — so the generic
# `Base.copyto!(::AbstractArray, ::Any)` fallback tries to iterate the tangent and
# fails with a `MethodError` for `iterate`. Bridge both Tangent shapes that arise.

# (a) Flat-Array-backed CV: tangent_type is
#     `Tangent{@NamedTuple{data::Vector{P}, axes::NoTangent}}`.
function Base.copyto!(
        dest::ComponentArray{P, N, <:Array{P}},
        src::Mooncake.Tangent{@NamedTuple{data::A, axes::Mooncake.NoTangent}},
    ) where {P <: _FloatLike, N, A <: Array{P}}
    copyto!(getdata(dest), src.fields.data)
    return dest
end

# (b) SubArray-backed CV (from `getproperty(::ComponentVector, ::Symbol)` on a nested
#     parent): tangent_type nests an inner Tangent that mirrors the SubArray's fields.
#     Symmetric to the `_increment_subarray_fdata!` path already in this file: copy is
#     only well-defined when the view fully covers its parent, since the SubArray
#     indices are not recoverable from Mooncake fdata/tangent shape alone.
function Base.copyto!(
        dest::ComponentArray{P, N, <:AbstractArray{P}},
        src::Mooncake.Tangent{
            @NamedTuple{
                data::Mooncake.Tangent{
                    @NamedTuple{
                        parent::Array{P, 1},
                        indices::Mooncake.NoTangent,
                        offset1::Mooncake.NoTangent,
                        stride1::Mooncake.NoTangent,
                    },
                },
                axes::Mooncake.NoTangent,
            },
        },
    ) where {P <: _FloatLike, N}
    parent = src.fields.data.fields.parent
    if length(parent) != length(getdata(dest))
        throw(
            ArgumentError(
                "ComponentArraysMooncakeExt: cannot copy a SubArray-backed " *
                    "ComponentVector tangent (parent length $(length(parent))) into a " *
                    "ComponentArray destination of length $(length(getdata(dest))). This " *
                    "happens when a tangent flows out of a view that does not fully cover " *
                    "its parent; there is no way to recover the view indices from Mooncake " *
                    "tangent fields alone. Please file an issue against ComponentArrays.jl " *
                    "with a reproducer.",
            ),
        )
    end
    copyto!(getdata(dest), parent)
    return dest
end

# ComponentArray's own Mooncake tangent type, for flat-Array-backed ComponentArrays.
# SubArray-backed ones still use the generic derivation and the rules above.

const FlatComponentArray{T <: _FloatLike, N, Axes} = ComponentArray{
    T, N, Array{T, N}, Axes,
}

# A ComponentVector's tangent is just another ComponentVector.
Mooncake.@foldable Mooncake.tangent_type(::Type{P}) where {P <: FlatComponentArray} = P
Mooncake.@foldable Mooncake.tangent_type(::Type{P}, ::Type{Mooncake.NoRData}) where {P <: FlatComponentArray} = P

# CuArray gets this for free by being mutable. ComponentArray is immutable and falls
# through to the generic derivation instead, which breaks, so declare it directly.
Mooncake.@foldable Mooncake.fdata_type(::Type{P}) where {P <: FlatComponentArray} = P
Mooncake.@foldable Mooncake.rdata_type(::Type{P}) where {P <: FlatComponentArray} = Mooncake.NoRData

Mooncake.tangent(p::FlatComponentArray, ::Mooncake.NoRData) = p

function Mooncake.zero_tangent_internal(x::FlatComponentArray, dict::Mooncake.MaybeCache)
    haskey(dict, x) && return dict[x]::typeof(x)
    t = zero(x)
    dict[x] = t
    return t
end

function Mooncake.randn_tangent_internal(
        rng::AbstractRNG, x::FlatComponentArray, dict::Mooncake.MaybeCache
    )
    haskey(dict, x) && return dict[x]::typeof(x)
    t = zero(x)
    randn!(rng, getdata(t))
    dict[x] = t
    return t
end

function Mooncake.increment_internal!!(c::Mooncake.IncCache, x::A, y::A) where {A <: FlatComponentArray}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
Mooncake.TestUtils.__increment_should_allocate(::Type{<:FlatComponentArray}) = true
Mooncake.set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::FlatComponentArray) = (x .= 0; x)

# Used by Mooncake's finite-difference tests, not the real backward pass.
function Mooncake._add_to_primal_internal(
        c::Mooncake.MaybeCache, x::P, y::P, unsafe::Bool
    ) where {P <: FlatComponentArray}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[key] = x′
    return x′
end
function Mooncake.primal_to_tangent_internal!!(t, x::FlatComponentArray, c::Mooncake.MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= x
    return t
end
function Mooncake.tangent_to_primal_internal!!(x::FlatComponentArray, t, c::Mooncake.MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= t
    return x
end

function Mooncake._dot_internal(c::Mooncake.MaybeCache, x::P, y::P) where {P <: FlatComponentArray}
    key = (x, y)
    haskey(c, key) && return c[key]
    v = dot(getdata(x), getdata(y))
    c[key] = v
    return v
end
function Mooncake._scale_internal(c::Mooncake.MaybeCache, a::Float64, t::P) where {P <: FlatComponentArray}
    haskey(c, t) && return c[t]
    t′ = a .* t
    c[t] = t′
    return t′
end

# Our fdata isn't one of Mooncake's standard shapes, so the generic getfield/lgetfield
# rule doesn't apply. Modeled on Mooncake's own lgetfield rule for Memory. :axes has no
# real content; :data's fdata aliases the array in x's tangent, so accumulation into it
# just works through mutation, no pullback needed.
Mooncake.@is_primitive(
    Mooncake.MinimalCtx, Tuple{typeof(Mooncake.lgetfield), P, Val{S}} where {P <: FlatComponentArray, S}
)
function Mooncake.rrule!!(
        ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
        x::Mooncake.CoDual{P, P},
        ::Mooncake.CoDual{Val{S}},
    ) where {P <: FlatComponentArray, S}
    y = getfield(Mooncake.primal(x), S)
    dy = S === :axes ? Mooncake.NoFData() : getfield(x.dx, S)
    return Mooncake.CoDual(y, dy), Mooncake.NoPullback(ntuple(_ -> Mooncake.NoRData(), 3))
end
function Mooncake.frule!!(
        ::Mooncake.Dual{typeof(Mooncake.lgetfield)},
        x::Mooncake.Dual{P, P},
        ::Mooncake.Dual{Val{S}},
    ) where {P <: FlatComponentArray, S}
    y = getfield(Mooncake.primal(x), S)
    dy = S === :axes ? Mooncake.NoTangent() : getfield(Mooncake.tangent(x), S)
    return Mooncake.Dual(y, dy)
end

# lgetfield only covers reading fields back out. Without a matching _new_ rule, anything
# that reconstructs a FlatComponentArray during tracing (e.g. plain broadcasting) fails.
# _new_ is already a universal Mooncake primitive, so no @is_primitive needed here.
function Mooncake.rrule!!(
        ::Mooncake.CoDual{typeof(Mooncake._new_)},
        ::Mooncake.CoDual{Type{P}},
        data::Mooncake.CoDual,
        axes::Mooncake.CoDual,
    ) where {P <: FlatComponentArray}
    y = Mooncake._new_(P, Mooncake.primal(data), Mooncake.primal(axes))
    dy = Mooncake._new_(P, data.dx, Mooncake.primal(axes))
    return Mooncake.CoDual(y, dy), Mooncake.NoPullback(ntuple(_ -> Mooncake.NoRData(), 4))
end
function Mooncake.frule!!(
        ::Mooncake.Dual{typeof(Mooncake._new_)},
        ::Mooncake.Dual{Type{P}},
        data::Mooncake.Dual,
        axes::Mooncake.Dual,
    ) where {P <: FlatComponentArray}
    y = Mooncake._new_(P, Mooncake.primal(data), Mooncake.primal(axes))
    dy = Mooncake._new_(P, Mooncake.tangent(data), Mooncake.primal(axes))
    return Mooncake.Dual(y, dy)
end

# @from_rrule/@from_chainrules bridge. Our tangent is already array-shaped, so it's
# ChainRules' own natural tangent as-is.
Mooncake.to_cr_tangent(t::FlatComponentArray) = t

# Same bridge, other direction. The increment_and_get_rdata! rules above are for the old
# generic FData shape and don't match a bare FlatComponentArray.
function Mooncake.increment_and_get_rdata!(f::P, ::Mooncake.NoRData, t::P) where {P <: FlatComponentArray}
    Mooncake.increment!!(f, t)
    return Mooncake.NoRData()
end

# Used by Mooncake's test suite for aliasing checks. The generic path calls
# __get_data_field, which doesn't know our type, so override it directly like Memory
# and CuArray do. pointer_from_objref doesn't work on ComponentArray (immutable), so
# delegate to Array's own handling of `.data` instead.
function Mooncake.TestUtils.populate_address_map_internal(
        m::Mooncake.TestUtils.AddressMap, p::FlatComponentArray, t::FlatComponentArray
    )
    return Mooncake.TestUtils.populate_address_map_internal(m, getdata(p), getdata(t))
end
function Mooncake.__verify_fdata_value(::IdDict{Any, Nothing}, p::FlatComponentArray, f::FlatComponentArray)
    if size(p) != size(f)
        throw(Mooncake.InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end
    return nothing
end

function Mooncake.TestUtils.has_equal_data_internal(
        x::P, y::P, equal_undefs::Bool, d::IdDict{Any, Bool}
    ) where {P <: FlatComponentArray}
    return isapprox(getdata(x), getdata(y))
end

end
