function ChainRulesCore.rrule(
        ::typeof(getproperty), x::ComponentArray, s::Union{
            Symbol, Val,
        }
    )
    return getproperty(x, s), Δ -> getproperty_adjoint(ChainRulesCore.unthunk(Δ), x, s)
end

function getproperty_adjoint(Δ, x, s)
    zero_x = zero(similar(x, eltype(Δ)))
    zero_x = __setproperty!(zero_x, s, Δ)
    return (ChainRulesCore.NoTangent(), zero_x, ChainRulesCore.NoTangent())
end

# Composite NamedTuple/Tangent cotangents: fill a zero ComponentArray by field name.
function getproperty_adjoint(Δ::ChainRulesCore.Tangent{<:Any, <:NamedTuple}, x, s)
    return getproperty_adjoint(_fill_named_tangent(getproperty(x, s), Δ), x, s)
end

function getproperty_adjoint(Δ::NamedTuple, x, s)
    return getproperty_adjoint(_fill_named_tangent(getproperty(x, s), Δ), x, s)
end

function _tangent_eltype(Δ, fallback)
    for (_, v) in pairs(Δ)
        v isa ChainRulesCore.AbstractZero && continue
        if v isa AbstractArray
            return eltype(v)
        elseif v isa Number
            return typeof(v)
        elseif v isa NamedTuple || v isa ChainRulesCore.Tangent
            return _tangent_eltype(v, fallback)
        end
    end
    return fallback
end

function _fill_named_tangent!(z, Δ)
    for (k, v) in pairs(Δ)
        v isa ChainRulesCore.AbstractZero && continue
        if v isa NamedTuple || v isa ChainRulesCore.Tangent
            _fill_named_tangent!(getproperty(z, k), v)
        else
            setproperty!(z, k, v)
        end
    end
    return z
end

function _fill_named_tangent(template, Δ)
    T = _tangent_eltype(Δ, eltype(template))
    return _fill_named_tangent!(zero(similar(template, T)), Δ)
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

function ChainRulesCore.rrule(
        cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        ::typeof(__setproperty!), x, s, Δ
    )
    y_, pb_f = ChainRulesCore.rrule_via_ad(cfg, __setproperty!, Val(true), x, s, Δ)
    return y_, pb_f
end

function ChainRulesCore.rrule(::typeof(getdata), x::ComponentArray)
    return getdata(x),
        Δ -> (ChainRulesCore.NoTangent(), ComponentArray(ChainRulesCore.unthunk(Δ), getaxes(x)))
end

function ChainRulesCore.rrule(::Type{ComponentArray}, data, axes)
    return ComponentArray(data, axes),
        Δ -> (
            ChainRulesCore.NoTangent(), getdata(ChainRulesCore.unthunk(Δ)),
            ChainRulesCore.NoTangent(),
        )
end

function ChainRulesCore.ProjectTo(ca::ComponentArray)
    return ChainRulesCore.ProjectTo{ComponentArray}(;
        project = ChainRulesCore.ProjectTo(getdata(ca)), axes = getaxes(ca)
    )
end

function (p::ChainRulesCore.ProjectTo{ComponentArray})(dx::AbstractArray)
    return ComponentArray(p.project(dx), p.axes)
end

# Prevent double projection
(p::ChainRulesCore.ProjectTo{ComponentArray})(dx::ComponentArray) = dx

function (p::ChainRulesCore.ProjectTo{ComponentArray})(
        t::ChainRulesCore.Tangent{
            A, <:NamedTuple,
        }
    ) where {A}
    nt = Functors.fmap(ChainRulesCore.backing, ChainRulesCore.backing(t))
    return ComponentArray(nt)
end

function ChainRulesCore.rrule(::Type{CA}, nt::NamedTuple) where {CA <: ComponentArray}
    y = CA(nt)

    ∇NamedTupleToComponentArray(Δ) = ∇NamedTupleToComponentArray(ChainRulesCore.unthunk(Δ))

    function ∇NamedTupleToComponentArray(Δ::AbstractArray)
        if length(Δ) == length(y)
            return ∇NamedTupleToComponentArray(ComponentArray(vec(Δ), getaxes(y)))
        end
        error(
            "Got pullback input of shape $(size(Δ)) & type $(typeof(Δ)) for output " *
                "of shape $(size(y)) & type $(typeof(y))"
        )
        return nothing
    end

    function ∇NamedTupleToComponentArray(Δ::ComponentArray)
        return ChainRulesCore.NoTangent(), NamedTuple(Δ)
    end

    return y, ∇NamedTupleToComponentArray
end
