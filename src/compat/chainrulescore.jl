function ChainRulesCore.rrule(::typeof(getproperty), x::ComponentArray, s::Union{Symbol,Val})
    return getproperty(x, s), Δ -> getproperty_adjoint(ChainRulesCore.unthunk(Δ), x, s)
end

function getproperty_adjoint(Δ, x, s)
    zero_x = zero(similar(x, eltype(Δ)))
    zero_x = __setproperty!(zero_x, s, Δ)
    return (ChainRulesCore.NoTangent(), zero_x, ChainRulesCore.NoTangent())
end

__setproperty!(x, s, Δ) = __setproperty!(Val(false), x, s, Δ)
function __setproperty!(::Val{false}, x, s, Δ)
    setproperty!(x, s, Δ)
    return x
end
# NOTE: I am not sure how this is avoiding the problem of mutation but if we wrap the
#       mutating function into an `rrule` as done here, Zygote computes the correct
#       gradient.
__setproperty!(::Val{true}, x, s::Symbol, Δ) = __setproperty!(Val(true), x, Val(s), Δ)
function __setproperty!(::Val{true}, x, s::Val, Δ)
    setproperty!(x, s, Δ)
    return x
end

function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
    ::typeof(__setproperty!), x, s, Δ)
    y_, pb_f = ChainRulesCore.rrule_via_ad(cfg, __setproperty!, Val(true), x, s, Δ)
    return y_, pb_f
end

ChainRulesCore.rrule(::typeof(getdata), x::ComponentArray) = getdata(x), Δ -> (ChainRulesCore.NoTangent(), ComponentArray(ChainRulesCore.unthunk(Δ), getaxes(x)))

ChainRulesCore.rrule(::Type{ComponentArray}, data, axes) = ComponentArray(data, axes), Δ -> (ChainRulesCore.NoTangent(), getdata(ChainRulesCore.unthunk(Δ)), ChainRulesCore.NoTangent())

function ChainRulesCore.ProjectTo(ca::ComponentArray)
    return ChainRulesCore.ProjectTo{ComponentArray}(; project=ChainRulesCore.ProjectTo(getdata(ca)), axes=getaxes(ca))
end

(p::ChainRulesCore.ProjectTo{ComponentArray})(dx::AbstractArray) = ComponentArray(p.project(dx), p.axes)

# Prevent double projection
(p::ChainRulesCore.ProjectTo{ComponentArray})(dx::ComponentArray) = dx

function (p::ChainRulesCore.ProjectTo{ComponentArray})(t::ChainRulesCore.Tangent{A,<:NamedTuple}) where {A}
    nt = Functors.fmap(ChainRulesCore.backing, ChainRulesCore.backing(t))
    return ComponentArray(nt)
end

function ChainRulesCore.rrule(::Type{CA}, nt::NamedTuple) where {CA<:ComponentArray}
    y = CA(nt)

    ∇NamedTupleToComponentArray(Δ) = ∇NamedTupleToComponentArray(ChainRulesCore.unthunk(Δ))

    function ∇NamedTupleToComponentArray(Δ::AbstractArray)
        if length(Δ) == length(y)
            return ∇NamedTupleToComponentArray(ComponentArray(vec(Δ), getaxes(y)))
        end
        error("Got pullback input of shape $(size(Δ)) & type $(typeof(Δ)) for output " *
              "of shape $(size(y)) & type $(typeof(y))")
        return nothing
    end

    function ∇NamedTupleToComponentArray(Δ::ComponentArray)
        return ChainRulesCore.NoTangent(), NamedTuple(Δ)
    end

    return y, ∇NamedTupleToComponentArray
end
