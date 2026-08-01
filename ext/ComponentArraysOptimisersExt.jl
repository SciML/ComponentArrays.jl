module ComponentArraysOptimisersExt

using ComponentArrays
using Optimisers: Optimisers, AbstractRule

# Optimisers can handle componentarrays by default, but we can vectorize the entire
# operation here instead of doing multiple smaller operations
Optimisers.setup(opt::AbstractRule, ps::ComponentArray) = Optimisers.setup(opt, getdata(ps))

function Optimisers.update(tree, ps::ComponentArray, gs::ComponentArray)
    gs_flat = ComponentArrays.__value(getdata(gs)) # Safety against ReverseDiff
    tree, ps_new = Optimisers.update(tree, getdata(ps), gs_flat)
    return tree, ComponentArray(ps_new, getaxes(ps))
end

function Optimisers.update!(tree::Optimisers.Leaf, ps::ComponentArray, gs::ComponentArray)
    gs_flat = ComponentArrays.__value(getdata(gs)) # Safety against ReverseDiff
    tree, ps_new = Optimisers.update!(tree, getdata(ps), gs_flat)
    return tree, ComponentArray(ps_new, getaxes(ps))
end

end
