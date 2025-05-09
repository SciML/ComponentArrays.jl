# Show AbstractAxis instances
Base.show(io::IO, ::MIME"text/plain", ::Axis{IdxMap}) where IdxMap = print(io, "Axis$IdxMap")
Base.show(io::IO, ::Axis{IdxMap}) where IdxMap = print(io, "Axis$IdxMap")

Base.show(io::IO, ::FlatAxis) = print(io, "FlatAxis()")
Base.show(io::IO, ::MIME"text/plain", ::FlatAxis) = print(io, "FlatAxis()")

Base.show(io::IO, ::NullAxis) = print(io, "NullAxis()")

Base.show(io::IO, ::MIME"text/plain", ::PartitionedAxis{PartSz, IdxMap, Ax}) where {PartSz, IdxMap, Ax} =
    print(io, "PartitionedAxis($PartSz, $(Ax()))")
Base.show(io::IO, ::PartitionedAxis{PartSz, IdxMap, Ax}) where {PartSz, IdxMap, Ax} =
    print(io, "PartitionedAxis($PartSz, $(Ax()))")

Base.show(io::IO, ::ShapedAxis{Shape}) where {Shape} =
    print(io, "ShapedAxis($Shape)")
Base.show(io::IO, ::Shaped1DAxis{Shape}) where {Shape} =
    print(io, "Shaped1DAxis($Shape)")

Base.show(io::IO, ::MIME"text/plain", ::ViewAxis{Inds, IdxMap, Ax}) where {Inds, IdxMap, Ax} =
    print(io, "ViewAxis($Inds, $(Ax()))")
Base.show(io::IO, ::ViewAxis{Inds, IdxMap, <:Ax}) where {Inds, IdxMap, Ax} =
    print(io, "ViewAxis($Inds, $(Ax()))")
Base.show(io::IO, ::ViewAxis{Inds, IdxMap, <:NullorFlatAxis}) where {Inds, IdxMap} =
    print(io, Inds)

# Show reducted Axis types to avoid excessive stacktraces
Base.show(io::IO, ::Type{Axis{IdxMap}}) where {IdxMap} = print(io, "Axis{...}")
Base.show(io::IO, ::Type{FlatAxis}) = print(io, "FlatAxis")
Base.show(io::IO, ::Type{NullAxis}) = print(io, "NullAxis")
function Base.show(io::IO, ::Type{PartitionedAxis{PartSz,IdxMap,Ax}}) where {PartSz,IdxMap,Ax}
    return print(io, "PartitionedAxis{$PartSz, {...}, $Ax}")
end
Base.show(io::IO, ::Type{ShapedAxis{Shape,IdxMap}}) where {Shape,IdxMap} = print(io, "ShapedAxis($Shape, {...})")
Base.show(io::IO, ::Type{ViewAxis{Inds,IdxMap,Ax}}) where {Inds,IdxMap,Ax} = print(io, "ViewAxis{$Inds, {...}, $Ax)")

Base.show(io::IO, ci::ComponentIndex) = print(io, "ComponentIndex($(ci.idx), $(ci.ax))")




# Show ComponentArrays
_print_type_short(io, ca; color=:normal) = _print_type_short(io, typeof(ca); color=color)
_print_type_short(io, T::Type; color=:normal) = printstyled(io, T; color=color)
_print_type_short(io, ::Type{<:ComponentArray{T,N,<:Array}}; color=:normal) where {T,N} = printstyled(io, "ComponentArray{$T,$N}"; color=color) # do not pollute the stacktrace with verbose type printing
_print_type_short(io, ::Type{<:ComponentArray{T,1,<:Array}}; color=:normal) where {T} = printstyled(io, "ComponentVector{$T}"; color=color)
_print_type_short(io, ::Type{<:ComponentArray{T,2,<:Array}}; color=:normal) where {T} = printstyled(io, "ComponentMatrix{$T}"; color=color)
_print_type_short(io, ::Type{<:ComponentArray{T,N,<:SubArray}}; color=:normal) where {T,N} = printstyled(io, "ComponentArray{$T,$N,SubArray...}"; color=color) # do not pollute the stacktrace with verbose type printing
_print_type_short(io, ::Type{<:ComponentArray{T,1,<:SubArray}}; color=:normal) where {T} = printstyled(io, "ComponentVector{$T,SubArray...}"; color=color)
_print_type_short(io, ::Type{<:ComponentArray{T,2,<:SubArray}}; color=:normal) where {T} = printstyled(io, "ComponentMatrix{$T,SubArray...}"; color=color)

@static if v"1.6" ≤ VERSION < v"1.8"
    Base.print_type_stacktrace(io, CA::Type{<:ComponentArray}; color=:normal) = _print_type_short(io, CA; color=color)
end

function Base.show(io::IO, x::ComponentVector)
    print(io, "(")
    for (i,key) in enumerate(keys(x))
        if i==1
            print(io, "$key = ")
        else
            print(io, ", $key = ")
        end
        show(io, x[key])
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", x::ComponentVector)
    len = length(x)
    ax = getaxes(x)[1]
    if last_index(ax) == len
        _print_type_short(io, x)
        show(io, x)
    else
        print(io, "$len-element ")
        _print_type_short(io, x)
        println(io, " with axis $ax:")
        Base.print_array(io, getdata(x))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", x::ComponentMatrix{T,A,Axes}) where {T,A,Axes}
    if !haskey(io, :compact) && length(axes(x, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    axs = getaxes(x)
    sz = size(x)
    print(io, "$(sz[1])×$(sz[2]) ")
    _print_type_short(io, x)
    println(io, " with axes $(axs[1]) × $(axs[2])")
    Base.print_matrix(io, getdata(x))
    return nothing
end
