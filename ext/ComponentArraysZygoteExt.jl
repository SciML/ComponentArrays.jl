module ComponentArraysZygoteExt

using ComponentArrays, Zygote

# For most cases this work. However, if the ComponentArray contains ROCArray, it fails to
# compile the broadcast operation on AMDGPU. This will most likely be fixed with proper
# broadcast mechanics in AMDGPU.jl but we can work around that in a harmless fashion for
# now.
function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

function Zygote.seed(x::ComponentArray, ::Val{N}, offset = 0) where{N}
    data = Zygote.seed(getdata(x), Val(N), offset)

    ComponentArray(data, getaxes(x))
end

end
